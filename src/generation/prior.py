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
        """Forward pass of the denoiser network.

        Args:
            x: Input batch `(B, d)`.
            sigma: Noise level features `(B, 1)` used to build embeddings.

        Returns:
            Denoised batch `(B, d)`.
        """
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
    def __init__(self, settings, rng_key: jax.Array | None = None):
        """Data generator for high-resolution samples via forward SDE.

        Args:
            settings: Settings dict containing beta schedule, batch sizes, dt, T, d.
            rng_key: Optional PRNG key for reproducibility.
        """
        self.beta_min = float(settings["general"]["beta_min"])
        self.beta_max = float(settings["general"]["beta_max"])
        self.T = float(settings["general"]["T"])
        self.batch_size = int(settings["pre_trained"]["data"]["batch_size"])
        self.d = int(settings["general"]["d"])
        self.dt = float(settings["general"]["dt"])
        self.n_samples = int(settings["general"]["n_samples"])
        self.rng = rng_key if rng_key is not None else jax.random.PRNGKey(37)

    @staticmethod
    @partial(jax.jit, static_argnums=(2, 3, 4, 5, 6))
    def euler_maruyama_solver(
        key: jax.Array,
        x_0: jax.Array,
        b: Callable[[jax.Array, jax.Array], jax.Array],
        g: Callable[[jax.Array], jax.Array],
        dt: float,
        T: float,
        return_trajectory: bool = False,
    ) -> jax.Array:
        """Solver for the forward SDE using Euler-Maruyama.

        Args:
            key: Random seed.
            x_0: Initial condition.
            b: Drift coefficient.
            g: Diffusion coefficient.
            dt: Time step.
            T: Time horizon.

        Returns:
            Sample from the target distribution.
        """

        def time_step(carry: tuple[jax.Array, jax.Array], params_time):
            key, x = carry
            t, dt = params_time
            key, subkey = jax.random.split(key)
            diff = g(t)
            deltaW = jax.random.normal(key=subkey, shape=(x.shape))
            drift = b(x, t)
            x = x + dt * drift + jnp.sqrt(dt) * diff * deltaW
            return (key, x), (x)

        key, _ = jax.random.split(key)
        n_steps = int(1 / dt)
        time_steps = T * jnp.arange(1, n_steps) / (n_steps - 1)
        delta_t = time_steps[1:] - time_steps[:-1]
        params_time = jnp.stack([time_steps[:-1], delta_t], axis=1)
        carry = (key, x_0)
        (_, samples), trajectory = jax.lax.scan(time_step, carry, params_time)
        if return_trajectory:
            return trajectory
        else:
            return samples

    def p(self, x, t):
        """Toy target density used for sample generation.

        Args:
            x: State `(B, d)`.
            t: Time scalar or `(B, 1)`; unused in this example.

        Returns:
            Unnormalized density values `(B,)`.
        """
        return jnp.exp(-jnp.abs(jnp.sum(jnp.square(x)) - 1.0))

    def sigma_noise(self, t):
        """Noise magnitude schedule for the forward SDE.

        Args:
            t: Time array.

        Returns:
            Noise standard deviation at time `t`.
        """
        return 0.0 * t + jnp.sqrt(2) * 0.001

    def logp(self, x, t):
        """Log-density corresponding to `p(x, t)`.

        Args:
            x: State `(B, d)`.
            t: Time.

        Returns:
            Log-density values `(B,)`.
        """
        return jnp.log(self.p(x, t))

    def nablaV(self, x, t):
        """Score function ∇ₓ log p(x, t)."""
        return jax.grad(self.logp, argnums=0)(x, t)

    def get_samples(self):
        """Generate a batch of samples from the forward SDE.

        Returns:
            Array of shape `(n_samples, d)` with final-time samples.
        """
        sampler_vectorized = jax.vmap(
            partial(
                HR_data.euler_maruyama_solver,
                b=self.nablaV,
                g=self.sigma_noise,
                dt=self.dt,
                T=self.T,
            ),
            in_axes=(0, 0),
            out_axes=0,
        )
        key_array = jax.random.split(self.rng, self.n_samples)
        x_0 = jax.random.normal(self.rng, shape=(self.n_samples, self.d))
        samples = sampler_vectorized(key_array, x_0)
        return samples


class HR_prior:
    def __init__(self, samples, settings, rng_key: jax.Array | None = None):
        """Prior model trained to approximate the score of the data distribution.

        Args:
            samples: Training samples `(N, d)`.
            settings: Settings dict with model hyperparameters.
            rng_key: Optional PRNG key.
        """
        self.samples = samples
        self.d = settings["general"]["d"]
        self.dt = settings["general"]["dt"]
        self.T = settings["general"]["T"]
        self.beta_min = settings["general"]["beta_min"]
        self.beta_max = settings["general"]["beta_max"]
        self.N_epochs = settings["pre_trained"]["model"]["N_epochs"]
        self.batch_size = settings["pre_trained"]["model"]["batch_size"]
        self.train_size = samples.shape[0]
        self.steps_per_epoch = self.train_size // self.batch_size
        self.denoiser = Denoiser()
        self.rng = rng_key if rng_key is not None else jax.random.PRNGKey(0)
        # Define dummy data for initailization.
        self.x, self.time = jnp.zeros((self.batch_size, self.d)), jnp.ones(
            (self.batch_size, 1)
        )
        self.params = self.denoiser.init(self.rng, self.x, self.time)
        self.optimizer = optax.adam(1e-4)
        self.opt_state = self.optimizer.init(self.params)

    # f,g not necessary here but just to indicate where the s(), sigma2 come from.
    def f(self, x: jax.Array, t: jax.Array) -> jax.Array:
        """Drift for forward SDE: f(x, t) = -0.5 * β(t) * x."""
        return -0.5 * self.beta(t) * x

    def g(self, t: jax.Array) -> jax.Array:
        """Diffusion magnitude: g(t) = sqrt(β(t))."""
        return jnp.sqrt(self.beta(t))

    def beta(self, t: jax.Array) -> jax.Array:
        """Linear beta schedule between `beta_min` and `beta_max`."""
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def int_beta(self, t: jax.Array) -> jax.Array:
        """Integral of β(t) from 0 to t."""
        return t * self.beta_min + 0.5 * t**2 * (self.beta_max - self.beta_min)

    def s(self, t: jax.Array) -> jax.Array:
        """Scaling schedule s(t) = exp(-0.5 * ∫ beta)."""
        return jnp.exp(-0.5 * self.int_beta(t))

    def s2sigma2(self, t: jax.Array) -> jax.Array:
        """Helper returning s^2(t) = 1 - exp(-∫ beta)."""
        return 1 - jnp.exp(-self.int_beta(t))

    def sigma2(self, t: jax.Array) -> jax.Array:
        """Variance schedule sigma^2(t) = exp(∫ beta) - 1."""
        return jnp.exp(self.int_beta(t)) - 1

    def loss_fn(
        self, params: PyTree, model: nn.Module, rng: jax.Array, batch: jax.Array
    ) -> jax.Array:
        """Loss function for denoiser training.

        Args:
            params: Current weights of the model.
            model: The score function (denoiser) module.
            rng: JAX PRNG key.
            batch: A batch of samples `(B, d)`.

        Returns:
            Scalar loss value.
        """
        num_steps = int(self.T / self.dt)
        rng, step_rng = jax.random.split(rng)
        batch_size = batch.shape[0]
        t = jax.random.randint(step_rng, (batch_size, 1), 1, num_steps) / (
            num_steps - 1
        )
        # Extract the standard deviation from the schedule.
        std = jnp.sqrt(self.sigma2(t))
        rng, step_rng = jax.random.split(rng)
        noise = jax.random.normal(step_rng, batch.shape)
        xt = batch + noise * std
        output = model.apply(params, xt, std)
        loss = jnp.mean((batch - output) ** 2)
        return loss

    @partial(jax.jit, static_argnums=[0, 5])
    def update_step(
        self,
        params: PyTree,
        rng: jax.Array,
        batch: jax.Array,
        opt_state: optax.OptState,
        model: nn.Module,
    ) -> tuple[jax.Array, PyTree, optax.OptState]:
        """One optimizer update step for the denoiser.

        Computes the gradient of the loss and applies an optimizer step.

        Args:
            params: The current weights of the model.
            rng: Random seed from jax.
            batch: A batch of samples from the training data, representing samples
                from p_{\text{data}}, shape (d, N)
            opt_state: the internal state of the optimizer
            model: the score function

        Returns:
            Tuple `(loss, new_params, new_opt_state)`.
        """
        val, grads = jax.value_and_grad(self.loss_fn)(params, model, rng, batch)
        updates, opt_state = self.optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return val, params, opt_state

    def train(self, log_fn=None):
        """Train the prior model on samples.

        Args:
            log_fn: Optional callable to log metrics per epoch.
        """
        for k in tqdm(range(self.N_epochs)):
            self.rng, step_rng = jax.random.split(self.rng)
            # Permutes the data at each epoch, potentially change.
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
            if log_fn is not None:
                try:
                    log_fn(
                        {
                            "prior/epoch": int(k),
                            "prior/mean_loss": float(mean_loss),
                        }
                    )
                except Exception:
                    pass

    def trained_score(self, x, t):
        """Compute the learned score using the denoiser.

        Args:
            x: Inputs `(B, d)`.
            t: Time `(B, 1)`.

        Returns:
            Estimated score `(B, d)`.
        """
        x_t = x / self.s(t)
        std = jnp.sqrt(self.sigma2(t))

        return (self.s(t) * self.denoiser.apply(self.params, x_t, std) - x) / (
            self.s2sigma2(t) + 1e-6
        )
