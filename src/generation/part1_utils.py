import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import Callable
import numpy as np


def plot_samples(samples, output_dir, name):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], s=3, alpha=0.8)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Scatter Plot of Samples")
    plt.savefig(output_dir + name)


def get_summary(samples_before, samples_after, settings):
    """Evaluate how well the method samples from the conditional distribution p(x|Cx=y)."""

    C = np.array(settings["pde_solver"]["C"])
    y_before = samples_before @ C.T
    y_after = samples_after @ C.T
    constraint_error = y_after - y_before

    print(f"\n=== NEW SAMPLES CONSTRAINT ANALYSIS ===")

    print(f"New samples Cx values - Mean: {np.mean(y_after, axis=0)}")
    print(f"New samples Cx values - Std: {np.std(y_after, axis=0)}")
    print(
        f"New samples Cx values - Range: [{np.min(y_after, axis=0)}, {np.max(y_after, axis=0)}]"
    )

    training_y_mean = np.mean(y_before, axis=0)
    training_y_std = np.std(y_before, axis=0)
    print(f"Training y distribution - Mean: {training_y_mean}")
    print(f"Training y distribution - Std: {training_y_std}")

    z_scores = (y_after - training_y_mean) / (training_y_std + 1e-8)
    print(
        f"Z-scores of new samples: Mean={np.mean(z_scores):.3f}, Std={np.std(z_scores):.3f}"
    )
    print(f"Percentage of new samples within ±2σ: {np.mean(np.abs(z_scores) < 2):.3f}")

    norm_value = jnp.linalg.norm(constraint_error)
    print(f"\n=== NORM METRIC ===")
    print(f"Norm value: {norm_value}")

    return {
        "constraint_error_mean": float(np.mean(constraint_error)),
        "constraint_error_std": float(np.std(constraint_error)),
        "new_samples_y_mean": float(np.mean(y_after)),
        "new_samples_y_std": float(np.std(y_after)),
        "z_score_mean": float(np.mean(z_scores)),
        "z_score_std": float(np.std(z_scores)),
        "norm_value": float(norm_value),
    }


@partial(jax.jit, static_argnums=(1, 2, 3, 4, 5, 6, 7))
def sde_solver_backwards_cond(
    key: jax.Array,
    grad_log: Callable[[jax.Array], jax.Array],
    grad_log_h: Callable[[jax.Array], jax.Array],
    g: Callable[[jax.Array], jax.Array],
    f: Callable[[jax.Array, jax.Array], jax.Array],
    N: int,
    n_samples: int,
    sigma2: Callable[[jax.Array], jax.Array],
    ts: jax.Array = jnp.arange(1, 1000) / (1000 - 1),
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
        disp = g(1 - t)
        t = jnp.ones((x.shape[0], 1)) * t
        # Directly call JAX-native grad_log_h
        grad_log_h_result = grad_log_h(x, 1 - t)

        drift = (
            -f(x, 1 - t)
            + grad_log(x, 1 - t) * disp**2
            + 100 * grad_log_h_result * disp**2
        )
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
    x_1 = jnp.sqrt(sigma2(1.0)) * jax.random.normal(subkey, shape=(n_samples, N))
    carry = (key, x_1)
    (_, samples), _ = jax.lax.scan(lmc_step_with_kernel, carry, params_time)
    return x_1, samples
