from jax_cfd.base import advection
from jax_cfd.base import grids
import jax
from jax_cfd.base import grids

GridVariable = grids.GridVariable
GridArray = grids.GridArray
Array = grids.Array
from typing import Callable, Optional
from jax_cfd.base import advection
from data.jax_cfd_new_base.diffusion import solve_ks_fast_diag


def kuramoto_sivashinsky(
    nu: float,
    gamma: float,
    dt: float,
    grid: grids.Grid,
    convect: Optional[Callable[[GridVariable], GridArray]] = None,
    implicit_solve: Callable = solve_ks_fast_diag,
    forcing: Optional[Callable[[GridVariable], GridArray]] = None,
) -> Callable[[GridVariable], GridVariable]:
    """Returns a function that performs a time step of the KS equation."""

    if convect is None:

        def convect(u: GridVariable) -> GridArray:
            v_vector = (u,)
            advection_term = advection.advect_van_leer_using_limiters(u, v_vector, dt)
            return 0.5 * advection_term

    convect = jax.named_call(convect, name="convection")
    implicit_solve = jax.named_call(implicit_solve, name="implicit_solve")

    @jax.named_call
    def ks_step(u: GridVariable) -> GridVariable:
        """Computes state at time t + dt."""
        dudt_explicit = convect(u)

        if forcing is not None:
            dudt_explicit = dudt_explicit + forcing(u)

        # Perform arithmetic on raw .data arrays to avoid tracer type issues.
        u_star_data = u.data + dudt_explicit.data * dt
        # Manually re-wrap the result into a GridArray before creating the GridVariable.
        u_star_array = grids.GridArray(u_star_data, u.offset, grid)
        u_star = grids.GridVariable(u_star_array, u.bc)

        u_next = implicit_solve(u_star, nu, gamma, dt)
        return u_next

    return ks_step
