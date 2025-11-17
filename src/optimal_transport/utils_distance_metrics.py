import jax
import jax.numpy as jnp
from jax.scipy.stats import gaussian_kde
from jax.scipy.integrate import trapezoid
from functools import partial


def _single_dimension_calculate_kld(
    predicted_samples: jnp.ndarray,
    reference_samples: jnp.ndarray,
    epsilon: float = 1e-10,
) -> float:
    """
    Calculates the Kullback-Leibler divergence for a single dimension.
    """

    # Extract the 1D marginal data for the current dimension
    pred_data = jnp.squeeze(predicted_samples)
    ref_data = jnp.squeeze(reference_samples)

    # Create Kernel Density Estimations for both distributions
    kde_pred = gaussian_kde(pred_data, bw_method="scott")
    kde_ref = gaussian_kde(ref_data, bw_method="scott")

    # Create a common grid of points to evaluate the PDFs
    min_val = jnp.minimum(jnp.min(pred_data), jnp.min(ref_data))
    max_val = jnp.maximum(jnp.max(pred_data), jnp.max(ref_data))
    max_val = jnp.where(max_val == min_val, min_val + 1e-6, max_val)
    grid = jnp.linspace(min_val, max_val, 256)

    # Evaluate the PDFs on the grid
    pdf_pred = kde_pred(grid)
    pdf_ref = kde_ref(grid)

    # Calculate the integrand for KL divergence without boolean indexing
    mask = pdf_ref > epsilon
    integrand = jnp.where(mask, pdf_ref * jnp.log(pdf_ref / (pdf_pred + epsilon)), 0.0)

    # Approximate the integral using the trapezoidal rule over full grid
    kld_m = trapezoid(integrand, x=grid)

    return kld_m


@jax.jit
def _single_calculate_kld(
    predicted_samples: jnp.ndarray,
    reference_samples: jnp.ndarray,
    epsilon: float = 1e-10,
) -> float:
    """
    Calculates the aggregated Kullback-Leibler divergence using Kernel Density Estimation.

    This metric is defined in "Debias Coarsely, Sample Conditionally: Statistical
    Downscaling through Optimal Transport and Probabilistic Diffusion Models"
    (Supplementary Material, Appendix C, Eq. 35). It computes the KL divergence
    for the 1D marginal distributions of each dimension and sums them up.

    The formula is:
    KLD = sum_{m=1 to d} integral( p_m_ref(v) * log(p_m_ref(v) / p_m_pred(v)) dv )

    Where:
    - d is the number of dimensions (pixels).
    - p_m_ref and p_m_pred are the 1D marginal probability density functions
      (PDFs) for the m-th dimension, estimated using KDE with Scott's rule
      for bandwidth selection.
    - The integral is approximated using the trapezoidal rule.

    Args:
        predicted_samples (np.ndarray): A 2D numpy array of shape (N, d),
            representing the generated samples.
        reference_samples (np.ndarray): A 2D numpy array of shape (M, d),
            representing the ground truth or reference samples HFHR. N can be different from M.
        epsilon (float): A small value to add to the predicted PDF to avoid
            division by zero in the log.

    Returns:
        float: The total KLD over all dimensions.
    """
    if predicted_samples.shape[1] != reference_samples.shape[1]:
        raise ValueError(
            "Predicted and reference samples must have the same number of dimensions (columns)."
        )

    kld_vec = jax.vmap(
        _single_dimension_calculate_kld, in_axes=(1, 1, None), out_axes=0
    )(predicted_samples, reference_samples, epsilon)
    total_kld = jnp.sum(kld_vec)

    return total_kld


@jax.jit
def calculate_kld_pooled(
    predicted_samples: jnp.ndarray,
    reference_samples: jnp.ndarray,
    epsilon: float = 1e-10,
) -> float:
    num_pooled_samples = predicted_samples.shape[0] * predicted_samples.shape[1]
    num_dimensions = predicted_samples.shape[2]
    pooled_predicted_samples = jnp.reshape(
        predicted_samples,
        (num_pooled_samples, num_dimensions, predicted_samples.shape[3]),
    )

    num_pooled_samples = reference_samples.shape[0] * reference_samples.shape[1]
    num_dimensions = reference_samples.shape[2]
    pooled_reference_samples = jnp.reshape(
        reference_samples,
        (num_pooled_samples, num_dimensions, reference_samples.shape[3]),
    )
    kld = _single_calculate_kld(
        pooled_predicted_samples, pooled_reference_samples, epsilon
    )
    return kld


def calculate_kld_OT(policy_gradient, true_data_model):

    # Vectorized, key-driven sampling for B=100 trajectories from both models
    B = policy_gradient.run_sett["num_samples_metrics"]
    key_master = jax.random.PRNGKey(int(policy_gradient.run_sett["seed"]))
    key_model, key_true = jax.random.split(key_master)
    keys_model = jax.random.split(key_model, B)
    keys_true = jax.random.split(key_true, B)

    y_trajs, y_trajs_prime = jax.vmap(
        policy_gradient.normalized_flow_model.sample_trajectory, in_axes=(0,)
    )(keys_model)
    z_trajs, z_trajs_prime = jax.vmap(
        true_data_model.sample_true_trajectory, in_axes=(0,)
    )(keys_true)

    # Pooled KLD across batch and time axes, per channel
    kld_OT = calculate_kld_pooled(y_trajs, z_trajs)
    kld_OT_prime = calculate_kld_pooled(y_trajs_prime, z_trajs_prime)

    return (kld_OT, kld_OT_prime)


def _single_dimension_calculate_wass1(
    predicted_samples_1d: jnp.ndarray,
    reference_samples_1d: jnp.ndarray,
    num_bins: int = 1000,
) -> float:
    """
    Calculates the Wasserstein-1 metric for a single dimension using empirical CDFs.

    The integral is approximated over the fixed range [-20, 20] as specified.
    """
    # Define the integration range and bins
    integration_range = [
        -20.0,
        20.0,
    ]  # this can be changed to the range of the data!!! I think also sufficient for KS
    bins = jnp.linspace(integration_range[0], integration_range[1], num_bins + 1)

    # Ensure data is 1D for histogram
    pred_data = jnp.squeeze(predicted_samples_1d)
    ref_data = jnp.squeeze(reference_samples_1d)

    # Compute histograms (empirical PDFs)
    counts_pred, _ = jnp.histogram(pred_data, bins=bins, range=integration_range)
    counts_ref, _ = jnp.histogram(ref_data, bins=bins, range=integration_range)

    # Compute empirical CDFs. Add epsilon for numerical stability if sum is zero.
    total_pred = jnp.sum(counts_pred)
    total_ref = jnp.sum(counts_ref)

    cdf_pred = jnp.cumsum(counts_pred) / (total_pred + 1e-10)
    cdf_ref = jnp.cumsum(counts_ref) / (total_ref + 1e-10)

    # Calculate the absolute difference between CDFs
    cdf_diff = jnp.abs(cdf_pred - cdf_ref)

    # Get bin centers for trapezoidal integration
    # The CDF values correspond to the probability mass *within* each bin
    bin_centers = (bins[:-1] + bins[1:]) / 2.0

    # Approximate the integral using the trapezoidal rule
    # This is equivalent to sum(|CDF_pred - CDF_ref| * dz)
    wass1_m = trapezoid(cdf_diff, x=bin_centers)

    return wass1_m


@partial(jax.jit, static_argnames="num_bins")
def _single_calculate_wass1(
    predicted_samples: jnp.ndarray,
    reference_samples: jnp.ndarray,
    num_bins: int = 1000,
) -> float:
    """
    Calculates the aggregated Wasserstein-1 metric (Wass1) as defined in Eq. 38.

    It computes the Wass1 metric for the 1D marginal distributions of each
    dimension and averages them.

    The formula is:
    Wass1 = (1/d) * sum_{m=1 to d} integral( |CDF_pred,m(z) - CDF_ref,m(z)| dz )

    Args:
        predicted_samples (jnp.ndarray): Array of shape (N, d, 1),
            representing the generated samples.
        reference_samples (jnp.ndarray): Array of shape (M, d, 1),
            representing the reference samples. N can be different from M.
        num_bins (int): The number of bins to use for the empirical CDF
            and integration. Must be static for JIT.

    Returns:
        float: The mean Wass1 metric over all dimensions.
    """
    if predicted_samples.shape[1] != reference_samples.shape[1]:
        raise ValueError(
            "Predicted and reference samples must have the same number of dimensions (columns)."
        )
    if predicted_samples.shape[2] != 1 or reference_samples.shape[2] != 1:
        raise ValueError(
            f"Expected trailing dimension of 1, but got {predicted_samples.shape} and {reference_samples.shape}"
        )

    # Vectorize the 1D calculation over all dimensions (axis=1)
    # in_axes=(1, 1, None) maps over dimension 'd' for both samples
    # and passes the static 'num_bins' argument
    wass1_vec = jax.vmap(
        _single_dimension_calculate_wass1, in_axes=(1, 1, None), out_axes=0
    )(predicted_samples, reference_samples, num_bins)

    # Average the Wass1 metric across all dimensions
    mean_wass1 = jnp.mean(wass1_vec)

    return mean_wass1


@partial(jax.jit, static_argnames="num_bins")
def calculate_wass1_pooled(
    predicted_samples: jnp.ndarray,
    reference_samples: jnp.ndarray,
    num_bins: int = 1000,
) -> float:
    """
    JIT-compiled entry point for pooled Wass1 calculation.

    Pools the batch and condition axes of predicted_samples before metric calculation.

    Args:
        predicted_samples (jnp.ndarray): Shape (N, C, D, 1)
        reference_samples (jnp.ndarray): Shape (M, D, 1)
        num_bins (int): Number of bins for histogram.

    Returns:
        float: The mean Wass1 metric.
    """
    # Pool the N (batch) and C (condition) axes
    num_pooled_samples = predicted_samples.shape[0] * predicted_samples.shape[1]
    num_dimensions = predicted_samples.shape[2]
    pooled_predicted_samples = jnp.reshape(
        predicted_samples,
        (num_pooled_samples, num_dimensions, predicted_samples.shape[3]),
    )

    num_pooled_samples = reference_samples.shape[0] * reference_samples.shape[1]
    num_dimensions = reference_samples.shape[2]
    pooled_reference_samples = jnp.reshape(
        reference_samples,
        (num_pooled_samples, num_dimensions, reference_samples.shape[3]),
    )

    wass1 = _single_calculate_wass1(
        pooled_predicted_samples, pooled_reference_samples, num_bins
    )
    return wass1


def calculate_wass1_OT(policy_gradient, true_data_model):

    # Vectorized, key-driven sampling for B=100 trajectories from both models
    B = policy_gradient.run_sett["num_samples_metrics"]
    key_master = jax.random.PRNGKey(int(policy_gradient.run_sett["seed"]))
    key_model, key_true = jax.random.split(key_master)
    keys_model = jax.random.split(key_model, B)
    keys_true = jax.random.split(key_true, B)

    y_trajs, y_trajs_prime = jax.vmap(
        policy_gradient.normalized_flow_model.sample_trajectory, in_axes=(0,)
    )(keys_model)
    z_trajs, z_trajs_prime = jax.vmap(
        true_data_model.sample_true_trajectory, in_axes=(0,)
    )(keys_true)

    wass1_OT = calculate_wass1_pooled(y_trajs, y_trajs_prime)
    wass1_OT_prime = calculate_wass1_pooled(z_trajs, z_trajs_prime)

    return (wass1_OT, wass1_OT_prime)
