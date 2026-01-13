"""Utilities for evaluating and visualizing flow-based optimal transport models.

This module provides helper functions used during training/evaluation to:

- fetch evaluation parameters (e.g., EMA) from a policy/gradient object
- sample trajectories from the learned flow model
- compare marginal distributions at a specific time step between true data and flow
- compute adjacent-step correlations along trajectories
- plot correlation comparisons between flow-generated and true trajectories

Conventions
-----------
- A trajectory in the ORIGINAL data space has shape (B, N+1, D, 1), where:
  - B: batch size (number of trajectories)
  - N+1: number of time indices including n=0; correlations are over pairs (n, n-1)
  - D: data dimensionality
  - trailing singleton channel dimension

External interfaces
-------------------
- policy_gradient: object expected to implement
    - get_eval_params_trees() -> pytree of parameters used for evaluation
    - sample_trajectories(key, num, params) -> (y, y_prime) with shapes as above
- true_data_model: object expected to implement
    - sample_true_trajectory(key) -> (y, y_prime) single sample with shapes (N+1, D, 1)
- writer: object expected to implement
    - write_images(images: dict[str, str], step: Optional[int] = None) -> None
- run_sett: mapping (e.g., dict) providing at least the keys used in this module:
    - global.seed (int)
    - global.num_iterations (int)
    - metrics.num_samples (int)
    - metrics.plot_samples_max (int)
    - metrics.chunk_size (int)
    - metrics.adjcorr_max_samples (int)
    - work_dir (str, optional; defaults to current working directory)
"""

import jax
import jax.numpy as jnp
import os
import matplotlib.pyplot as plt


def _get_eval_params(policy_gradient):
    """Return the parameter pytree to be used for evaluation.

    If the training code maintains an exponential moving average (EMA) of the
    parameters, this function should return those EMA parameters via the
    `get_eval_params_trees` hook on the provided object.

    Parameters
    ----------
    policy_gradient
        An object exposing `get_eval_params_trees() -> pytree`.

    Returns
    -------
    Any
        A pytree of parameters suitable for evaluation/sampling.

    Raises
    ------
    ValueError
        If `policy_gradient` does not implement `get_eval_params_trees`.
    """
    if hasattr(policy_gradient, "get_eval_params_trees"):
        return policy_gradient.get_eval_params_trees()
    else:
        raise ValueError("PolicyGradient does not have get_eval_params_trees method")


def _sample_flow_trajs(policy_gradient, key, num, params):
    """Sample trajectories from the learned flow model.

    Parameters
    ----------
    policy_gradient
        An object exposing
        `sample_trajectories(key: PRNGKey, num: int, params: pytree) -> (y, y_prime)`.
    key : jax.random.PRNGKey
        Random key for sampling.
    num : int
        Number of trajectories to sample (batch size).
    params
        Parameter pytree returned by `_get_eval_params`.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        Tuple `(y, y_prime)` with shapes:
        - y:  (num, N+1, D, 1)
        - y': (num, N+1, D, 1)

    Raises
    ------
    ValueError
        If `policy_gradient` does not implement `sample_trajectories`.
    """
    num = int(num)
    if hasattr(policy_gradient, "sample_trajectories"):
        return policy_gradient.sample_trajectories(key, num=num, params=params)
    else:
        raise ValueError("PolicyGradient does not have sample_trajectories method")


def plot_comparison(
    n,
    dims,
    policy_gradient,
    true_data_model,
    run_sett,
    writer,
    step=None,
    key_suffix: str = "",
):
    """Plot and save histogram comparisons at time step `n` for each dimension.

    For each dimension d in [0, dims), plot the marginal distributions of `y_n[d]`
    and `y'_n[d]` for both the true data process and the flow-generated trajectories.
    The figure is saved under `<work_dir>/hist/` and logged via `writer.write_images`.

    Parameters
    ----------
    n : int
        Time index at which to compare marginals (0 <= n <= N).
    dims : int
        Number of data dimensions D to visualize (assumes the first `dims` dims).
    policy_gradient
        Flow model provider with `get_eval_params_trees` and `sample_trajectories`.
    true_data_model
        Ground-truth process provider with `sample_true_trajectory(key)`.
    run_sett : Mapping
        Settings dictionary. Uses:
        - global.seed
        - metrics.num_samples
        - metrics.plot_samples_max
        - metrics.chunk_size
        - global.num_iterations (for default logging step)
        - work_dir (optional; defaults to CWD)
    writer
        Logger with `write_images(images: dict[str, str], step: Optional[int])`.
    step : Optional[int]
        Training step to associate with the logged image. If None, defaults to
        `global.num_iterations - 1` when logging.
    key_suffix : str
        Extra suffix appended to the saved filename and logging key (useful when
        producing multiple variants).

    Side Effects
    ------------
    - Saves a PNG to `<work_dir>/hist/comparison_hist_t{n}_dims{dims}_seed{...}{suffix}[_stepX].png`.
    - Logs the image path via `writer.write_images`.
    """
    num_bins = 50
    key = jax.random.PRNGKey(int(run_sett["global"]["seed"]))
    B_plot = int(
        min(run_sett["metrics"]["num_samples"], run_sett["metrics"]["plot_samples_max"])
    )
    params = _get_eval_params(policy_gradient)

    true_y_list, true_y_prime_list = [], []
    flow_y_list, flow_y_prime_list = [], []

    remaining = B_plot
    cur_key = key
    chunk = int(run_sett["metrics"]["chunk_size"])

    while remaining > 0:
        cur = min(remaining, chunk)
        cur_key, key_true, key_flow = jax.random.split(cur_key, 3)

        keys_true = jax.random.split(key_true, cur)
        z, zp = jax.vmap(true_data_model.sample_true_trajectory)(keys_true)

        y, yp = _sample_flow_trajs(policy_gradient, key_flow, cur, params)

        true_y_list.append(z)
        true_y_prime_list.append(zp)
        flow_y_list.append(y)
        flow_y_prime_list.append(yp)

        remaining -= cur

    true_y = jnp.concatenate(true_y_list, axis=0)
    true_y_prime = jnp.concatenate(true_y_prime_list, axis=0)
    flow_y = jnp.concatenate(flow_y_list, axis=0)
    flow_y_prime = jnp.concatenate(flow_y_prime_list, axis=0)

    t_y_n = true_y[:, n, :, 0]
    f_y_n = flow_y[:, n, :, 0]
    t_yp_n = true_y_prime[:, n, :, 0]
    f_yp_n = flow_y_prime[:, n, :, 0]

    fig, axes = plt.subplots(dims, 2, figsize=(12, 4 * dims))
    if dims == 1:
        axes = axes.reshape(1, 2)

    for d in range(dims):
        ax_y = axes[d, 0]
        data_y = jnp.concatenate([t_y_n[:, d], f_y_n[:, d]])
        bins_y = jnp.linspace(data_y.min(), data_y.max(), num_bins)
        ax_y.hist(t_y_n[:, d], bins=bins_y, alpha=0.5, density=True, label="True Data")
        ax_y.hist(
            f_y_n[:, d], bins=bins_y, alpha=0.5, density=True, label="Flow Generated"
        )
        ax_y.set_title(f"y_{d} (t={n})")
        ax_y.set_ylabel("Density")
        ax_y.legend()

        ax_yp = axes[d, 1]
        data_yp = jnp.concatenate([t_yp_n[:, d], f_yp_n[:, d]])
        bins_yp = jnp.linspace(data_yp.min(), data_yp.max(), num_bins)
        ax_yp.hist(
            t_yp_n[:, d], bins=bins_yp, alpha=0.5, density=True, label="True Data"
        )
        ax_yp.hist(
            f_yp_n[:, d], bins=bins_yp, alpha=0.5, density=True, label="Flow Generated"
        )
        ax_yp.set_title(f"y'_{d} (t={n})")
        ax_yp.legend()

    fig.tight_layout()
    base_dir = run_sett.get("work_dir", os.getcwd())
    out_dir = os.path.join(base_dir, "hist")
    os.makedirs(out_dir, exist_ok=True)
    step_suffix = f"_step{int(step)}" if step is not None else ""
    out_path = os.path.join(
        out_dir,
        f"comparison_hist_t{n}_dims{dims}_seed{run_sett.get('seed', 'unknown')}{key_suffix}{step_suffix}.png",
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    try:
        target_step = (
            int(step)
            if step is not None
            else int(run_sett["global"]["num_iterations"]) - 1
        )
        writer.write_images(
            images={"comparison_hist" + str(n) + key_suffix: out_path}, step=target_step
        )
    except Exception:
        writer.write_images(images={"comparison_hist" + str(n) + key_suffix: out_path})


def _adjacent_corr_from_trajs(trajs: jnp.ndarray, eps: float = 1e-10):
    """Compute adjacent-step correlations along trajectories.

    For each time index n in [1, N], compute the correlation between y_n and
    y_{n-1} across the batch for each dimension, then average over dimensions.

    Parameters
    ----------
    trajs : jnp.ndarray
        Tensor of shape (B, N+1, D, 1) in the ORIGINAL space.
    eps : float, default 1e-10
        Small constant to stabilize division by near-zero standard deviations.

    Returns
    -------
    jnp.ndarray
        Array of shape (N,) where entry n-1 is the mean over dimensions of
        Corr(y_n, y_{n-1}) computed across the batch.
    """
    y = trajs[..., 0]  # (B, N+1, D)
    _, N_len, _ = y.shape
    corr_list = []
    for n in range(1, N_len):
        y_prev = y[:, n - 1, :]
        y_curr = y[:, n, :]
        y_prev_c = y_prev - jnp.mean(y_prev, axis=0, keepdims=True)
        y_curr_c = y_curr - jnp.mean(y_curr, axis=0, keepdims=True)
        cov = jnp.mean(y_prev_c * y_curr_c, axis=0)
        std_prev = jnp.sqrt(jnp.mean(y_prev_c**2, axis=0) + eps)
        std_curr = jnp.sqrt(jnp.mean(y_curr_c**2, axis=0) + eps)
        corr_dim = cov / (std_prev * std_curr + eps)
        corr_list.append(jnp.mean(corr_dim))
    return jnp.array(corr_list)


def calculate_adjacent_corr(class_instance, model_type: str, key):
    """Calculate adjacent-step correlations for either flow or true trajectories.

    Parameters
    ----------
    class_instance
        If `model_type == "flow"`, this is the flow object exposing
        `get_eval_params_trees` and `sample_trajectories`. Otherwise, it should
        expose `sample_true_trajectory(key)`.
    model_type : str
        Either `"flow"` or "true". When `"flow"`,
        samples are drawn from the learned model; otherwise from the true model.
    key : jax.random.PRNGKey
        Random key used for sampling.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        `(corr, corr_prime)` where each is shape (N,) with mean over dims of
        Corr(y_n, y_{n-1}) computed across the batch, for y and y' respectively.
    """
    run_sett = class_instance.run_sett
    B = int(
        min(
            run_sett["metrics"]["num_samples"],
            run_sett["metrics"]["adjcorr_max_samples"],
        )
    )
    chunk_size = int(run_sett["metrics"]["chunk_size"])
    params = _get_eval_params(class_instance) if model_type == "flow" else None
    y_list, yp_list = [], []
    remaining = B
    cur_key = key
    while remaining > 0:
        cur = min(remaining, chunk_size)
        cur_key, use_key = jax.random.split(cur_key)
        if model_type == "flow":
            y_trajs, y_trajs_prime = _sample_flow_trajs(
                class_instance, use_key, cur, params
            )
        else:
            keys_true = jax.random.split(use_key, cur)
            y_trajs, y_trajs_prime = jax.vmap(class_instance.sample_true_trajectory)(
                keys_true
            )
        y_list.append(y_trajs)
        yp_list.append(y_trajs_prime)
        remaining -= cur

    y_trajs = jnp.concatenate(y_list, axis=0)
    y_trajs_prime = jnp.concatenate(yp_list, axis=0)

    corr = _adjacent_corr_from_trajs(y_trajs)
    corr_prime = _adjacent_corr_from_trajs(y_trajs_prime)
    return corr, corr_prime


def plot_adjacent_corrs(
    corr_flow,
    corr_flow_prime,
    corr_true,
    corr_true_prime,
    run_sett,
    writer,
    first_k,
    step=None,
    key_suffix: str = "",
):
    """Plot and save adjacent-step correlation comparisons.

    Plots the first `first_k` entries of the correlation curves for the flow and
    true processes, for both y and y'. The figure is saved under
    `<work_dir>/adjcorr/` and logged via `writer.write_images`.

    Parameters
    ----------
    corr_flow : jnp.ndarray
        Shape (N,) correlation curve for y from the flow model.
    corr_flow_prime : jnp.ndarray
        Shape (N,) correlation curve for y' from the flow model.
    corr_true : jnp.ndarray
        Shape (N,) correlation curve for y from the true process.
    corr_true_prime : jnp.ndarray
        Shape (N,) correlation curve for y' from the true process.
    run_sett : Mapping
        Settings dictionary. Uses:
        - global.num_iterations (for default logging step)
        - work_dir (optional; defaults to CWD)
    writer
        Logger with `write_images(images: dict[str, str], step: Optional[int])`.
    first_k : int
        Number of initial correlation steps to plot (min with available length).
    step : Optional[int]
        Training step to associate with the logged image. If None, defaults to
        `global.num_iterations - 1` when logging.
    key_suffix : str
        Extra suffix appended to the saved filename and logging key.

    Side Effects
    ------------
    - Saves a PNG to `<work_dir>/adjcorr/adjcorr_comparison{suffix}[_stepX].png`.
    - Logs the image path via `writer.write_images`.
    """
    max_k = min(first_k, len(corr_flow))
    xs = jnp.arange(1, max_k + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(
        xs, corr_flow[:max_k], marker="o", linestyle="-", label="Flow Corr(y_t,y_{t-1})"
    )
    plt.plot(
        xs,
        corr_flow_prime[:max_k],
        marker="s",
        linestyle="--",
        label="Flow Corr(y'_t,y'_{t-1})",
    )
    plt.plot(
        xs, corr_true[:max_k], marker="o", linestyle="-", label="True Corr(y_t,y_{t-1})"
    )
    plt.plot(
        xs,
        corr_true_prime[:max_k],
        marker="s",
        linestyle="--",
        label="True Corr(y'_t,y'_{t-1})",
    )

    plt.title(f"Adjacent-step Correlation Comparison (first {max_k} steps)")
    plt.xlabel("t (correlation between t and t-1)")
    plt.ylabel("Correlation")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    base_dir = run_sett.get("work_dir", os.getcwd())
    out_dir = os.path.join(base_dir, "adjcorr")
    os.makedirs(out_dir, exist_ok=True)
    step_suffix = f"_step{int(step)}" if step is not None else ""
    out_path = os.path.join(out_dir, f"adjcorr_comparison{key_suffix}{step_suffix}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    try:
        target_step = (
            int(step)
            if step is not None
            else int(run_sett["global"]["num_iterations"]) - 1
        )
        writer.write_images(
            images={"adjcorr_comparison" + key_suffix: out_path}, step=target_step
        )
    except Exception:
        writer.write_images(images={"adjcorr_comparison" + key_suffix: out_path})
