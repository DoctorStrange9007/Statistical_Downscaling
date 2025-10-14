import matplotlib.pyplot as plt
import os
import jax
import jax.numpy as jnp
from functools import partial
from typing import Callable
import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity


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
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    plt.savefig(os.path.join(output_dir, name))


def plot_marginals_1_2_and_joint12(samples, output_dir, names, settings):
    """Save smooth marginals (X1, X3) and joint (X1, X3) using KDE.

    Args:
        samples: Array of shape `(N, d>=3)`.
        output_dir: Directory to save figures.
        names: List/tuple of three file names: [x1.png, x3.png, joint.png].
    """
    os.makedirs(output_dir, exist_ok=True)
    arr = np.asarray(samples)
    x_1 = arr[:, 0].astype(np.float64)
    x_3 = arr[:, 2].astype(np.float64)
    # Filter non-finite values before any KDE/hist operations
    mask = np.isfinite(x_1) & np.isfinite(x_3)
    x_1 = x_1[mask]
    x_3 = x_3[mask]
    if x_1.size == 0 or x_3.size == 0:
        return

    # 1) KDE marginal of X1 using scikit-learn KernelDensity
    fig1, ax1 = plt.subplots(figsize=(5, 4))
    grid1 = np.linspace(np.percentile(x_1, 0.5), np.percentile(x_1, 99.5), 800)
    bw1 = max(1e-6, 1.06 * np.std(x_1) * (x_1.size ** (-1 / 5)))
    kde1 = KernelDensity(kernel="gaussian", bandwidth=bw1).fit(x_1.reshape(-1, 1))
    d1 = np.exp(kde1.score_samples(grid1.reshape(-1, 1)))
    ax1.plot(grid1, d1, color="C0", label="KDE X1")
    ax1.fill_between(grid1, d1, color="C0", alpha=0.2)
    ax1.set_title("KDE of X1")
    ax1.set_xlabel("X1")
    ax1.axvline(
        float(settings["pde_solver"]["y_target"][0]),
        color="red",
        linestyle="--",
        linewidth=2,
        label="y_1",
    )
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(os.path.join(output_dir, names[0]), dpi=200)
    plt.close(fig1)

    # 2) KDE marginal of X3 using scikit-learn KernelDensity
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    grid3 = np.linspace(np.percentile(x_3, 0.5), np.percentile(x_3, 99.5), 800)
    bw3 = max(1e-6, 1.06 * np.std(x_3) * (x_3.size ** (-1 / 5)))
    kde3 = KernelDensity(kernel="gaussian", bandwidth=bw3).fit(x_3.reshape(-1, 1))
    d3 = np.exp(kde3.score_samples(grid3.reshape(-1, 1)))
    ax2.plot(grid3, d3, color="C1", label="KDE X3")
    ax2.fill_between(grid3, d3, color="C1", alpha=0.2)
    ax2.set_title("KDE of X3")
    ax2.set_xlabel("X3")
    ax2.axvline(
        float(settings["pde_solver"]["y_target"][1]),
        color="red",
        linestyle="--",
        linewidth=2,
        label="y_2",
    )
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, names[1]), dpi=200)
    plt.close(fig2)

    # 3) 3D joint density (X1, X3) using 2D KernelDensity with standardization
    fig3 = plt.figure(figsize=(7, 5))
    ax3 = fig3.add_subplot(111, projection="3d")
    x1_min, x1_max = np.percentile(x_1, [1, 99])
    x3_min, x3_max = np.percentile(x_3, [1, 99])
    x1_grid = np.linspace(x1_min, x1_max, 120)
    x3_grid = np.linspace(x3_min, x3_max, 120)
    X, Y = np.meshgrid(x1_grid, x3_grid)

    Zdata = np.stack([x_1, x_3], axis=1)
    mean_vec = Zdata.mean(axis=0)
    std_vec = Zdata.std(axis=0)
    std_vec[std_vec == 0.0] = 1.0
    Zstd = (Zdata - mean_vec) / std_vec
    n = Zstd.shape[0]
    bw2 = max(1e-6, 1.06 * (n ** (-1 / 6)))
    kde2 = KernelDensity(kernel="gaussian", bandwidth=bw2).fit(Zstd)

    grid_points = np.stack([X.ravel(), Y.ravel()], axis=1)
    grid_points_std = (grid_points - mean_vec) / std_vec
    d2_std = np.exp(kde2.score_samples(grid_points_std)).reshape(X.shape)
    jac = float(std_vec[0] * std_vec[1])
    Z = d2_std / jac

    ax3.plot_surface(X, Y, Z, cmap="viridis", linewidth=0, antialiased=True)
    ax3.set_title("Joint density of (X1, X3)")
    ax3.set_xlabel("X1")
    ax3.set_ylabel("X3")
    ax3.set_zlabel("density")
    fig3.tight_layout()
    fig3.savefig(os.path.join(output_dir, names[2]), dpi=200)
    plt.close(fig3)


@partial(jax.jit, static_argnums=(1, 2, 3, 4, 5, 6, 7, 8, 9, 11))
def sde_solver_backwards_cond(
    key: jax.Array,
    grad_log: Callable[[jax.Array], jax.Array],
    grad_log_h: Callable[[jax.Array], jax.Array],
    g: Callable[[jax.Array], jax.Array],
    f: Callable[[jax.Array, jax.Array], jax.Array],
    d: int,
    n_samples: int,
    T: float,
    sigma2: Callable[[jax.Array], jax.Array],
    s: Callable[[jax.Array], jax.Array],
    ts: jax.Array = jnp.arange(1, 1000) / (1000 - 1),
    conditional: bool = True,
) -> tuple[jax.Array, jax.Array]:
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
    Returns:
        Tuple `(x_T, samples)` where:
        - `x_T`: Terminal initialization `(n_samples, d)` drawn with std `sqrt(sigma2(T))`.
        - `samples`: Sequence of samples over time with final shape `(n_steps-1, n_samples, d)`.
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
    # Enforce equal scaling in 2D
    ax.axis("equal")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    return fig_path
