from jax_cfd.base.initial_conditions import wrap_variables
from jax_cfd.base import grids
from jax_cfd.base import boundaries
import jax
import jax.numpy as jnp

GridVariable = grids.GridVariable
GridArray = grids.GridArray


def kuramoto_sivashinsky_initial_conditions(
    grid: grids.Grid,
    key: grids.Array,
    n_c: int = 30,
) -> GridVariable:
    """Creates and wraps initial conditions for the Kuramoto-Sivashinsky equation.

    This function implements the random sum of sines specified in Eq. (44) of
    the reference paper and wraps it as a GridVariable at the cell center with
    periodic boundary conditions.

    Args:
      grid: The grid on which the field is defined.
      key: A JAX PRNG key for generating random numbers.
      n_c: The number of cosine terms to sum (default is 30).

    Returns:
      A GridVariable representing the initial state u(x, 0).
    """
    x = grid.axes()[0]
    L = grid.domain[0][1]

    # Split the PRNG key for each random parameter
    a_key, omega_key, phi_key = jax.random.split(key, 3)

    # Generate random amplitudes, frequencies, and phases
    a = jax.random.uniform(a_key, shape=(n_c,), minval=-0.5, maxval=0.5)
    omega_choices = jnp.array([2 * jnp.pi / L, 4 * jnp.pi / L, 6 * jnp.pi / L])
    omega = jax.random.choice(omega_key, omega_choices, shape=(n_c,))
    phi = jax.random.uniform(phi_key, shape=(n_c,), minval=0.0, maxval=2.0 * jnp.pi)

    # Construct the sum of sines
    sines = a[:, None] * jnp.sin(omega[:, None] * x[None, :] + phi[:, None])
    u0_data = jnp.sum(sines, axis=0)

    # Define boundary conditions and offsets for a cell-centered scalar
    bcs = (boundaries.periodic_boundary_conditions(grid.ndim),)

    # Use the wrap_variables helper to create the GridVariable
    # The comma unpacks the single-element tuple that is returned.
    (u_initial,) = wrap_variables(var=(u0_data,), grid=grid, bcs=bcs)
    return u_initial
