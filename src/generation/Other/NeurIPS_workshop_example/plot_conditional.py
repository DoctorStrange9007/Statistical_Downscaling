import yaml
import os
import sys
import argparse
import jax
import jax.numpy as jnp
from functools import partial
from typing import Callable

# Ensure repository root on path when running this file directly by path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.generation.Other.NeurIPS_workshop_example import utils_generation as utils


class HR_data_conditional:
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

    def p(self, x_2, t):
        """Toy target density used for sample generation.

        Args:
            x: State `(B, d)`.
            t: Time scalar or `(B, 1)`; unused in this example.

        Returns:
            Unnormalized density values `(B,)`.
        """
        return jnp.exp(-jnp.abs(jnp.square(x_2) - 0.75))

    def sigma_noise(self, t):
        """Noise magnitude schedule for the forward SDE.

        Args:
            t: Time array.

        Returns:
            Noise standard deviation at time `t`.
        """
        return 0.0 * t + jnp.sqrt(2) * 0.001

    def logp(self, x_2, t):
        """Log-density corresponding to `p(x, t)`.

        Args:
            x: State `(d)`.
            t: Time.

        Returns:
            Log-density values `(B,)`.
        """
        return jnp.log(self.p(x_2, t))

    def nablaV(self, x_2, t):
        """Score function ∇ₓ log p(x, t)."""
        return jax.grad(self.logp, argnums=0)(x_2, t)

    def get_samples(self):
        """Generate a batch of samples from the forward SDE.

        Returns:
            Array of shape `(n_samples, 3)` with final-time samples `(x1, x2, x3)`.
        """
        sampler_vectorized = jax.vmap(
            partial(
                HR_data_conditional.euler_maruyama_solver,
                b=self.nablaV,
                g=self.sigma_noise,
                dt=self.dt,
                T=self.T,
            ),
            in_axes=(0, 0),
            out_axes=0,
        )
        key_array = jax.random.split(self.rng, self.n_samples)
        x_0 = jax.random.normal(self.rng, shape=(self.n_samples))
        samples_x_2 = sampler_vectorized(key_array, x_0)
        samples_x_1 = jnp.full_like(samples_x_2, 0.5)
        samples_x_3 = jnp.zeros_like(samples_x_2)
        samples = jnp.stack([samples_x_1, samples_x_2, samples_x_3], axis=1)
        return samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="src/generation/settings_generation.yaml"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        run_sett = yaml.safe_load(f)
    if not os.path.exists(run_sett["output_dir"]):
        os.makedirs(run_sett["output_dir"])

    # Master RNG: use general.seed if present; else fixed
    seed = run_sett.get("general", {}).get("seed", 37)
    master_key = jax.random.PRNGKey(seed)
    key_data, key_prior, key_pde, key_sde = jax.random.split(master_key, 4)

    hr_data = HR_data_conditional(run_sett, rng_key=key_data)
    samples = hr_data.get_samples()
    utils.plot_samples(samples, run_sett["output_dir"], "true_conditional_samples.png")
