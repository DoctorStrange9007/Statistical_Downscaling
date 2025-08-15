import jax
import jax.numpy as jnp
from functools import partial

try:
    from .PDE_solver import PDE_solver
except ImportError:
    from PDE_solver import PDE_solver


def make_heat_settings(d: int, T: float, sampling_stages: int = 50) -> dict:
    """Minimal settings required by the base solver for the heat PDE.

    Args:
        d: Spatial dimension.
        T: Final time horizon.
        sampling_stages: Number of outer sampling stages used during training.

    Returns:
        Settings dictionary consumable by `PDE_solver` and `HeatPDESolver`.
    """
    return {
        "general": {"d": d, "T": T},
        "pde_solver": {
            "t_low": 1e-8,
            "x_low": -1.0,
            "x_high": 1.0,
            "x_multiplier": 1.0,
            "nSim_interior": 2048,
            "nSim_terminal": 2048,
            "sampling_stages": sampling_stages,
            "steps_per_sample": 10,
            "learning_rate": 1e-3,
        },
        # Network size
        "pre_trained": {"model": {"nodes_per_layer": 64, "num_layers": 3}},
    }


class HeatPDESolver(PDE_solver):
    """Backward heat equation in d dimensions with constant diffusion.

    ∂V/∂t + 0.5 * σ^2 * ΔV = 0,    V(T, x) = exp(a^T x)
    Exact solution: V(t, x) = exp(a^T x + 0.5 * σ^2 * ||a||^2 * (T - t))
    """

    def __init__(
        self,
        a_vec: jnp.ndarray,
        sigma: float,
        settings: dict,
        rng_key: jax.Array | None = None,
    ):
        """Initialize the heat PDE solver.

        Args:
            a_vec: Weight vector of shape `(d,)`.
            sigma: Constant diffusion coefficient.
            settings: Settings dictionary produced by `make_heat_settings` or similar.
            rng_key: Optional PRNG key for deterministic initialization.
        """
        d = int(settings["general"]["d"])
        assert a_vec.shape == (d,), "a_vec must have shape (d,)"
        self.a_vec = a_vec
        self.sigma = float(sigma)
        super().__init__(settings=settings, rng_key=rng_key)

    @partial(jax.jit, static_argnums=0)
    def loss_fn(self, params, t_interior, x_interior, t_terminal, x_terminal):
        """Compute PDE interior residual loss and terminal loss.

        Returns:
            Tuple `(L1, L3, total)` where `L1` is interior PDE residual MSE,
            `L3` is terminal MSE, and `total = L1 + L3`.
        """

        # Per-sample scalar V for gradient/hessian computations
        def V_single(ts: jax.Array, xs: jax.Array) -> jax.Array:
            ts_b = ts.reshape(1, 1)
            xs_b = xs.reshape(1, -1)
            return self.net.apply(params, ts_b, xs_b).squeeze()

        dV_dt_fn = jax.grad(lambda ts, xs: V_single(ts, xs), argnums=0)
        H_fn = jax.hessian(lambda xs, ts: V_single(ts, xs))

        V_t = jax.vmap(lambda ts, xs: dV_dt_fn(ts.squeeze(), xs))(
            t_interior, x_interior
        )
        V_xx = jax.vmap(lambda ts, xs: H_fn(xs, ts.squeeze()))(t_interior, x_interior)

        laplacian_x = jax.vmap(lambda m: jnp.trace(m))(V_xx)
        PDE_residual = (V_t.reshape(-1) + 0.5 * self.sigma**2 * laplacian_x).reshape(
            -1, 1
        )
        L1 = jnp.mean(jnp.square(PDE_residual))

        V_term = self.net.apply(params, t_terminal, x_terminal)
        target = jnp.exp(x_terminal @ self.a_vec.reshape(-1, 1))
        L3 = jnp.mean(jnp.square(V_term - target))
        return L1, L3, (L1 + L3)

    @partial(jax.jit, static_argnums=0)
    def exact_solution(self, t: jax.Array, x: jax.Array) -> jax.Array:
        """Closed-form solution to the backward heat equation for this setup.

        Args:
            t: Time tensor `(B, 1)`.
            x: Spatial tensor `(B, d)`.

        Returns:
            Exact solution `V(t, x)` as `(B, 1)` tensor.
        """
        aTx = x @ self.a_vec.reshape(-1, 1)
        return jnp.exp(
            aTx + 0.5 * (self.sigma**2) * (jnp.sum(self.a_vec**2)) * (self.T - t)
        )
