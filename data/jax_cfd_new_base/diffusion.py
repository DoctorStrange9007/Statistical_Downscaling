from jax_cfd.base.diffusion import _rhs_transform
from typing import Optional
from jax_cfd.base import boundaries
from jax_cfd.base import array_utils
from jax_cfd.base import fast_diagonalization
from jax_cfd.base import grids

GridVariable = grids.GridVariable
GridArray = grids.GridArray


def solve_ks_fast_diag(
    u: GridVariable,
    nu: float,
    gamma: float,
    dt: float,
    implementation: Optional[str] = None,
) -> GridVariable:
    """Solves the implicit part of the KS equation using fast diagonalization."""

    # The eigenvalue transformation for the KS implicit operator.
    # x is an eigenvalue of the Laplacian (∂_xx), hence, x^2 is an eigenvalue of the Laplacian squared (∂_xxxx).
    def func(x):
        # Eigenvalue of the combined operator (ν Δt ∂_xx - γ Δt ∂_xxxx)
        dt_nu_x = (dt * nu) * x
        dt_gamma_x_sq = (dt * gamma) * (x**2)
        numerator = dt_nu_x - dt_gamma_x_sq
        # We use the same identity u_new = u + [op/(1-op)]u for numerical stability.
        return numerator / (1 - numerator)

    if boundaries.has_all_periodic_boundary_conditions(u):
        circulant = True
    else:
        circulant = False
        implementation = "matmul"

    laplacians = array_utils.laplacian_matrix_w_boundaries(u.grid, u.offset, u.bc)

    ks_op = fast_diagonalization.transform(
        func,
        laplacians,
        u.dtype,
        hermitian=True,
        circulant=circulant,
        implementation=implementation,
    )

    u_interior = u.bc.trim_boundary(u.array)
    u_interior_transformed = _rhs_transform(u_interior, u.bc)
    u_dt_solved = grids.GridArray(
        ks_op(u_interior_transformed), u_interior.offset, u_interior.grid
    )

    u_solved_interior = u_interior + u_dt_solved
    u_final = u.bc.pad_and_impose_bc(u_solved_interior, offset_to_pad_to=u.offset)

    return u_final
