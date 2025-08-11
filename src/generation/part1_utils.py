import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from functools import partial
from typing import Callable


def plot_samples(samples, output_dir, name):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], s=3, alpha=0.8)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Scatter Plot of Samples")
    plt.savefig(output_dir + name)


@partial(jax.jit, static_argnums=(1, 2, 3, 4, 5, 6, 7, 8, 9, 11))
def sde_solver_backwards_cond(
    key: jax.Array,
    grad_log: Callable[[jax.Array], jax.Array],
    grad_log_h: Callable[[jax.Array], jax.Array],
    g: Callable[[jax.Array], jax.Array],
    f: Callable[[jax.Array, jax.Array], jax.Array],
    d: int,
    n_samples: int,
    T: int,
    sigma2: Callable[[jax.Array], jax.Array],
    s: Callable[[jax.Array], jax.Array],
    ts: jax.Array = jnp.arange(1, 1000) / (1000 - 1),
    conditional: bool = True,
) -> jax.Array:
    """Euler-Maruyama solver for the backwards SDE.

    Args:
        key: random number generator
        grad_log: drift term for the SDE (here the score)
        N: dimension of the problem
        dt: time step
        x_0: initial point
    """

    def lmc_step_with_kernel(carry, params_time):
        key, x = carry
        t, dt = params_time
        key, subkey = jax.random.split(key)
        # Now we run the kernel using Euler-Maruyama
        disp = g(T - t)
        t = jnp.ones((x.shape[0], 1)) * t
        # Directly call JAX-native grad_log_h
        grad_log_h_result = grad_log_h(x, t)

        if conditional:
            drift = (
                -f(x, T - t)
                + grad_log(x, T - t) * disp**2
                + grad_log_h_result * disp**2
            )
            x = (
                x
                + dt * drift
                + jnp.sqrt(dt) * disp * jax.random.normal(key=subkey, shape=(x.shape))
            )
        else:
            drift = -f(x, T - t) + grad_log(x, T - t) * disp**2
            x = (
                x
                + dt * drift
                + jnp.sqrt(dt) * disp * jax.random.normal(key=subkey, shape=(x.shape))
            )
        return (key, x), ()

    key, subkey = jax.random.split(key)
    dts = ts[1:] - ts[:-1]
    params_time = jnp.stack([ts[:-1], dts], axis=1)
    # Sampling the terminal condition with the correct std.
    x_1 = jnp.sqrt(sigma2(T)) * jax.random.normal(subkey, shape=(n_samples, d))
    carry = (key, x_1)
    (_, samples), _ = jax.lax.scan(lmc_step_with_kernel, carry, params_time)
    return x_1, samples
