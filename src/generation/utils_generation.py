import matplotlib.pyplot as plt
import os
import jax
import jax.numpy as jnp
from functools import partial
from typing import Callable
import pandas as pd
import numpy as np


def plot_samples(samples, output_dir, name):
    """Create and save a 3D scatter plot of samples.

    Args:
        samples: Array of shape `(N, 3)` with sample coordinates.
        output_dir: Directory to write the PNG into.
        name: File name of the PNG (no path joining performed here).
    """
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
        key: Random number generator key.
        grad_log: Drift term for the SDE (the score) mapping `(x, t) -> grad log p`.
        grad_log_h: Conditional drift component from PDE value function.
        g: Diffusion function.
        f: Drift function.
        d: Problem dimension.
        n_samples: Number of samples to draw.
        T: Time horizon.
        sigma2: Variance schedule.
        s: Scaling schedule.
        ts: Time grid in (0, T].
        conditional: If True, includes `grad_log_h` in the drift.
    """

    def lmc_step_with_kernel(carry, params_time):
        key, x = carry
        t, dt = params_time
        key, subkey = jax.random.split(key)
        # Now we run the kernel using Euler-Maruyama
        disp = g(T - t)
        t = jnp.ones((x.shape[0], 1)) * t
        # Compute conditional drift component if available; else zero
        if conditional and (grad_log_h is not None):
            grad_log_h_result = grad_log_h(x, t)
        else:
            grad_log_h_result = jnp.zeros_like(x)

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


def calculate_msd(samples_after, settings):
    """Compute mean squared distance from linear constraints Cx=y.

    Args:
        samples_after: Array `(N, d)` of samples.
        settings: Settings containing `pde_solver.C` and `pde_solver.y_target`.

    Returns:
        Scalar mean squared distance.
    """

    diff_after = samples_after @ jnp.array(settings["pde_solver"]["C"]).T - jnp.array(
        settings["pde_solver"]["y_target"]
    )
    mae_to_target_after = jnp.mean(jnp.square(diff_after).reshape(-1, 1))

    return mae_to_target_after


def output_to_excel_and_plot(all_msd, settings):
    """Write MSD results to Excel and save a line plot.

    Args:
        all_msd: Mapping of step/key (e.g., lambda) to scalar MSD.
        settings: Settings dict (used for `output_dir`).
    """
    # Ensure output directory exists
    output_dir = settings.get("output_dir", "output/")
    os.makedirs(output_dir, exist_ok=True)

    # Convert mapping to two-column DataFrame robustly (works for scalar values)
    keys = list(all_msd.keys())
    values = list(all_msd.values())

    # Sort by key if keys are comparable; otherwise keep insertion order
    try:
        sorted_pairs = sorted(zip(keys, values), key=lambda kv: kv[0])
        keys, values = zip(*sorted_pairs)
    except TypeError:
        pass

    df = pd.DataFrame({"step": keys, "msd": values})

    excel_path = os.path.join(output_dir, "part1_msd.xlsx")
    png_path = os.path.join(output_dir, "part1_msd.png")

    df.to_excel(excel_path, index=False, engine="openpyxl")

    # Plot: baseline as horizontal line, other entries as (lambda, msd) points
    fig, ax = plt.subplots(figsize=(10, 8))

    # Detect baseline value (support both 'input_data' and 'input data')
    baseline_value = None
    for baseline_key in ("input_data", "input data", "baseline", "input"):
        if baseline_key in all_msd:
            try:
                baseline_value = float(all_msd[baseline_key])
            except Exception:
                baseline_value = None
            break

    # Detect optional 'gen_without_conditioning' reference value
    gen_wo_cond_value = None
    if "gen_without_conditioning" in all_msd:
        try:
            gen_wo_cond_value = float(all_msd["gen_without_conditioning"])
        except Exception:
            gen_wo_cond_value = None

    # Collect numeric lambda points
    lambda_points = []
    for k, v in all_msd.items():
        if k in (
            "input_data",
            "input data",
            "baseline",
            "input",
            "gen_without_conditioning",
        ):
            continue
        try:
            x_val = float(k)
            y_val = float(v)
            lambda_points.append((x_val, y_val))
        except Exception:
            continue

    lambda_points.sort(key=lambda t: t[0])
    if lambda_points:
        xs, ys = zip(*lambda_points)
        ax.plot(xs, ys, marker="o", label="lambda sweep")

    if baseline_value is not None:
        ax.axhline(
            baseline_value, color="red", linestyle="--", linewidth=2, label="input data"
        )
    if gen_wo_cond_value is not None:
        ax.axhline(
            gen_wo_cond_value,
            color="green",
            linestyle="-.",
            linewidth=2,
            label="gen without conditioning",
        )

    ax.set_xlabel("lambda")
    ax.set_ylabel("msd")
    ax.set_title("Mean squared distance to y")
    ax.grid(True, alpha=0.3)
    if lambda_points or baseline_value is not None or gen_wo_cond_value is not None:
        ax.legend()
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    return {"excel": excel_path, "figure": png_path}


def plot_from_excel(settings):
    # Read the Excel produced by output_to_excel_and_plot
    output_dir = settings.get("output_dir", "output/")
    excel_path = os.path.join(output_dir, "all_data.xlsx")
    df = pd.read_excel(excel_path, engine="openpyxl")

    # Normalize column names
    df = df.rename(columns={"step": "step", "msd": "msd"})

    # Extract reference lines
    baseline_value = None
    for name in ("input_data", "input data", "baseline", "input"):
        if (df["step"] == name).any():
            try:
                baseline_value = float(df.loc[df["step"] == name, "msd"].iloc[0])
            except Exception:
                baseline_value = None
            break

    gen_wo_cond_value = None
    if (df["step"] == "gen_without_conditioning").any():
        try:
            gen_wo_cond_value = float(
                df.loc[df["step"] == "gen_without_conditioning", "msd"].iloc[0]
            )
        except Exception:
            gen_wo_cond_value = None

    # Extract numeric lambda points
    numeric_steps = pd.to_numeric(df["step"], errors="coerce")
    mask = numeric_steps.notna()
    xs = numeric_steps[mask].astype(float).to_numpy()
    ys = df.loc[mask, "msd"].astype(float).to_numpy()
    order = xs.argsort()
    xs, ys = xs[order], ys[order]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    if xs.size:
        ax.plot(xs, ys, marker="o", label="lambda sweep")
    if baseline_value is not None:
        ax.axhline(
            baseline_value, color="red", linestyle="--", linewidth=2, label="input data"
        )
    if gen_wo_cond_value is not None:
        ax.axhline(
            gen_wo_cond_value,
            color="green",
            linestyle="-.",
            linewidth=2,
            label="gen without conditioning",
        )

    ax.set_xlabel("lambda")
    ax.set_ylabel("msd")
    ax.set_title("Mean squared distance to y (from Excel)")
    ax.grid(True, alpha=0.3)
    if xs.size or (baseline_value is not None) or (gen_wo_cond_value is not None):
        ax.legend()
    fig.tight_layout()

    out_path = os.path.join(output_dir, "part1_msd_from_excel.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_hyperplane(samples, msd, settings, name, lambda_=None):
    projected_samples = samples @ jnp.array(settings["pde_solver"]["C"]).T
    # Convert to numpy for plotting
    proj_np = np.asarray(projected_samples)

    # Prepare output
    output_dir = settings.get("output_dir", "output/")
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, name)

    # Plot: x_2 on x-axis, x_1 on y-axis
    fig, ax = plt.subplots(figsize=(8, 6))
    if proj_np.shape[1] < 2:
        raise ValueError(
            "Projected samples must have at least 2 dimensions to plot (got {}).".format(
                proj_np.shape[1]
            )
        )

    ax.scatter(proj_np[:, 0], proj_np[:, 1], s=8, alpha=0.7, label="projected samples")

    # y_target in difference space is at (0, 0)
    ax.scatter(
        settings["pde_solver"]["y_target"][0],
        settings["pde_solver"]["y_target"][1],
        color="red",
        s=60,
        marker="x",
        label="y_target",
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    if lambda_ is not None:
        ax.set_title(
            "Projected samples (difference to y_target = "
            + str(msd)
            + ") for lambda = "
            + str(lambda_)
        )
    else:
        ax.set_title(
            "Projected samples (difference to y_target = "
            + str(msd)
            + ") without conditioning"
        )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    return fig_path
