import jax
import jax.numpy as jnp
import optax
from functools import partial
from tqdm import tqdm
from typing import Callable, Any
import flax.linen as nn

PyTree = Any


class Denoiser(nn.Module):
    """A simple model MLPs and a Fourier Embedding."""

    @nn.compact
    def __call__(self, x, sigma):
        in_size = x.shape[1]
        n_hidden = 256
        sigma = jnp.concatenate(
            [
                sigma,
                jnp.cos(2 * jnp.pi * sigma),
                jnp.sin(2 * jnp.pi * sigma),
                jnp.cos(4 * jnp.pi * sigma),
                jnp.sin(4 * jnp.pi * sigma),
            ],
            axis=1,
        )
        x = jnp.concatenate([x, sigma], axis=1)
        x = nn.Dense(n_hidden)(x)
        x = nn.selu(x)
        x = nn.Dense(n_hidden)(x)
        x = nn.selu(x)
        x = nn.Dense(n_hidden)(x)
        x = nn.selu(x)
        x = nn.Dense(in_size)(x)
        return x


class HR_data:
    def __init__(self, settings):
        self.beta_min = float(settings["general"]["beta_min"])
        self.beta_max = float(settings["general"]["beta_max"])
        self.T = float(settings["general"]["T"])
        self.batch_size = int(settings["pre_trained"]["data"]["batch_size"])
        self.d = int(settings["general"]["d"])
        self.dt = float(settings["general"]["dt"])

    @staticmethod
    @partial(jax.jit, static_argnums=(2, 3, 4, 5, 6))
    def euler_maruyama_solver(
        key: jax.Array,
        x_0: jax.Array,
        f: Callable[[jax.Array, jax.Array], jax.Array],
        g: Callable[[jax.Array], jax.Array],
        dt: float,
        t_final: float,
        return_trajectory: bool = False,
    ) -> jax.Array:
        """Solver for the forward equation using Euler-Maruyama.

        Args:
            key: Random seed.
            x_0: Initial condition.
            f: Drift coefficient.
            g: Diffusion coefficient.
            dt: Time step.
            t_final: Time horizon.

        Returns:
            Sample from the target distribution.
        """

        def time_step(carry: tuple[jax.Array, jax.Array], params_time):
            key, x = carry
            t, dt = params_time
            key, subkey = jax.random.split(key)
            # Now we run the kernel using Euler-Maruyama
            diff = g(t)
            deltaW = jax.random.normal(key=subkey, shape=(x.shape))
            drift = f(x, t)
            x = x + dt * drift + jnp.sqrt(dt) * diff * deltaW
            return (key, x), (x)

        key, _ = jax.random.split(key)
        n_steps = int(1 / dt)
        time_steps = t_final * jnp.arange(1, n_steps) / (n_steps - 1)
        # We allow for non-uniform discretization in time.
        delta_t = time_steps[1:] - time_steps[:-1]
        params_time = jnp.stack([time_steps[:-1], delta_t], axis=1)
        # Initial condition.
        carry = (key, x_0)
        (_, samples), trajectory = jax.lax.scan(time_step, carry, params_time)
        # Whether to return the full trajectory of just the last element.
        if return_trajectory:
            return trajectory
        else:
            return samples

    # Function proportional to the target distribution.
    def p(self, x, t):
        return jnp.exp(-jnp.abs(jnp.sum(jnp.square(x)) - 1.0)) + 0.0 * jnp.sum(t)

    # Noise level.
    def sigma_noise(self, t):
        return 0.0 * t + jnp.sqrt(2) * 0.001

    def logp(self, x, t):
        return jnp.log(self.p(x, t))

    # Compute the gradient of the log-likely hood or score function.
    def nablaV(self, x, t):
        return jax.grad(self.logp, argnums=0)(x, t)

    def get_samples(self):
        sampler_vectorized = jax.vmap(
            partial(
                HR_data.euler_maruyama_solver,
                f=self.nablaV,
                g=self.sigma_noise,
                dt=self.dt,
                t_final=self.T,
            ),
            in_axes=(0, 0),
            out_axes=0,
        )
        key = jax.random.PRNGKey(37)
        # Defines the number of samples.
        n_samples = 2000
        # It creates the array of random seeds.
        key_array = jax.random.split(key, n_samples)
        # We sample from the initial condition.
        x_0 = jax.random.normal(key, shape=(n_samples, self.d))
        # Sampling by solving the Langevin equation.
        samples = sampler_vectorized(key_array, x_0)
        return samples


class HR_prior:
    def __init__(self, samples, settings):
        self.samples = samples
        self.d = settings["general"]["d"]
        self.beta_min = settings["general"]["beta_min"]
        self.beta_max = settings["general"]["beta_max"]
        self.N_epochs = settings["pre_trained"]["model"]["N_epochs"]
        self.batch_size = settings["pre_trained"]["model"]["batch_size"]
        self.train_size = samples.shape[0]
        self.steps_per_epoch = self.train_size // self.batch_size
        self.denoiser = Denoiser()
        self.rng = jax.random.PRNGKey(0)
        # Define dummy data for initailization.
        self.x, self.time = jnp.zeros((self.batch_size, self.d)), jnp.ones(
            (self.batch_size, 1)
        )
        self.params = self.denoiser.init(self.rng, self.x, self.time)
        self.optimizer = optax.adam(1e-4)
        self.opt_state = self.optimizer.init(self.params)

    def beta(self, t: jax.Array) -> jax.Array:
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def int_beta(self, t: jax.Array) -> jax.Array:
        """Integral of beta from 0 to t."""
        return t * self.beta_min + 0.5 * t**2 * (self.beta_max - self.beta_min)

    def f(self, x: jax.Array, t: jax.Array) -> jax.Array:
        return -0.5 * self.beta(t) * x

    def g(self, t: jax.Array) -> jax.Array:
        return jnp.sqrt(self.beta(t))

    def s(self, t: jax.Array) -> jax.Array:
        return jnp.exp(-0.5 * self.int_beta(t))

    # Noise level.
    def sigma_noise(self, t):
        return 0.0 * t + jnp.sqrt(2) * 0.001

    def sigma2(self, t: jax.Array) -> jax.Array:
        return 1 - jnp.exp(-self.int_beta(t))

    @partial(jax.jit, static_argnums=[0, 5])
    def update_step(
        self,
        params: PyTree,
        rng: jax.Array,
        batch: jax.Array,
        opt_state: optax.OptState,
        model: nn.Module,
    ) -> tuple[jax.Array, PyTree, optax.OptState]:
        """Updates the parameters based on the gradient of the loss function.

        Takes the gradient of the loss function and updates the model weights
        (params) using it.

        Args:
            params: The current weights of the model.
            rng: Random seed from jax.
            batch: A batch of samples from the training data, representing samples
                from p_{\text{data}}, shape (d, N)
            opt_state: the internal state of the optimizer
            model: the score function

        Returns:
        A tuple with the value of the loss function (for metrics), the new parameters
        and the new
            optimizer state
        """
        val, grads = jax.value_and_grad(self.loss_fn)(params, model, rng, batch)
        updates, opt_state = self.optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return val, params, opt_state

    def loss_fn(
        self, params: PyTree, model: nn.Module, rng: jax.Array, batch: jax.Array
    ) -> jax.Array:
        """loss function to be used to train the model

        Args:
            params: the current weights of the model
            model: the score function
            rng: random number generator from jax
            batch: a batch of samples from the training data, representing samples
                from mu_text{data}, shape (J, N)

        Returns:
            The value of the loss function defined above.
        """
        rng, step_rng = jax.random.split(rng)
        batch_size = batch.shape[0]
        num_steps = 100
        t = jax.random.randint(step_rng, (batch_size, 1), 1, num_steps) / (
            num_steps - 1
        )
        # Extract the standard deviation from the schedule.
        std = jnp.sqrt(self.sigma_noise(t))
        rng, step_rng = jax.random.split(rng)
        noise = jax.random.normal(step_rng, batch.shape)
        xt = batch + noise * std
        output = model.apply(params, xt, std)
        loss = jnp.mean((batch - output) ** 2)
        return loss

    def train(self):
        for k in tqdm(range(self.N_epochs)):
            self.rng, step_rng = jax.random.split(self.rng)
            # Permutes the data at each epoch.
            data = jax.random.permutation(step_rng, self.samples, axis=0)
            data = jnp.reshape(
                data, (self.steps_per_epoch, self.batch_size, data.shape[-1])
            )
            losses = []
            for batch in data:
                self.rng, step_rng = jax.random.split(self.rng)
                loss, self.params, self.opt_state = self.update_step(
                    self.params, step_rng, batch, self.opt_state, self.denoiser
                )
                losses.append(loss)
            mean_loss = jnp.mean(jnp.array(losses))
            if k % 1000 == 0:
                print("Epoch %d \t, Mean Loss %f " % (k, mean_loss))

    def trained_score(self, x, t):
        x_t = x / self.s(t)

        return (
            self.s(t) * self.denoiser.apply(self.params, x_t, jnp.sqrt(self.sigma2(t)))
            - x
        ) / (self.sigma2(t) + 1e-6)
