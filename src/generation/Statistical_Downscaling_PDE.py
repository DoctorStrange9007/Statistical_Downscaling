import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

try:
    from .PDE_solver import PDE_solver
except ImportError:
    from PDE_solver import PDE_solver


class StatisticalDownscalingPDESolver(PDE_solver):
    """PDE for statistical downscaling with linear constraint Cx=y.

    Implements the loss used in the previous monolithic solver, including
    the reverse-time SDE drift components through grad_log and beta schedule.
    """

    def __init__(
        self,
        grad_log,
        samples: np.ndarray,
        settings: dict,
        rng_key: jax.Array | None = None,
    ):
        super().__init__(settings=settings, rng_key=rng_key)

        # Problem-specific attributes
        self.grad_log = grad_log  # Callable: (x, t) -> grad log p(x, t)
        self.C = jnp.array(settings["pde_solver"]["C"], dtype=jnp.float32)
        # Training-time y extracted from samples
        self.y = jnp.array(
            samples @ np.array(settings["pde_solver"]["C"]).T, dtype=jnp.float32
        )

        # Beta schedule parameters
        self.beta_min = float(settings["general"]["beta_min"])
        self.beta_max = float(settings["general"]["beta_max"])

    # Diffusion schedule and drifts for the OU-like SDE
    def beta(self, t: jax.Array) -> jax.Array:
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def f(self, x: jax.Array, t: jax.Array) -> jax.Array:
        return -0.5 * self.beta(t) * x

    def g(self, t: jax.Array) -> jax.Array:
        return jnp.sqrt(self.beta(t))

    def b_bar(self, t: jax.Array, x: jax.Array) -> jax.Array:
        disp = self.g(self.T - t)
        return -self.f(x, self.T - t) + self.grad_log(x, self.T - t) * (disp**2)

    @partial(jax.jit, static_argnums=0)
    def loss_fn(self, params, t_interior, x_interior, t_terminal, x_terminal):
        # Per-sample scalar V for gradient/hessian computations
        def V_single(ts: jax.Array, xs: jax.Array) -> jax.Array:
            ts_b = ts.reshape(1, 1)
            xs_b = xs.reshape(1, -1)
            return self.net.apply(params, ts_b, xs_b).squeeze()

        # Gradients and Hessians
        dV_dt_fn = jax.grad(lambda ts, xs: V_single(ts, xs), argnums=0)
        dV_dx_fn = jax.grad(lambda xs, ts: V_single(ts, xs))
        H_fn = jax.hessian(lambda xs, ts: V_single(ts, xs))

        V_t = jax.vmap(lambda ts, xs: dV_dt_fn(ts.squeeze(), xs))(
            t_interior, x_interior
        )
        V_x = jax.vmap(lambda ts, xs: dV_dx_fn(xs, ts.squeeze()))(
            t_interior, x_interior
        )
        V_xx = jax.vmap(lambda ts, xs: H_fn(xs, ts.squeeze()))(t_interior, x_interior)

        # b_bar and diffusion trace term
        bbar = self.b_bar(t_interior, x_interior)
        disp2 = (self.g(self.T - t_interior) ** 2).reshape(-1)
        trace_term = jax.vmap(lambda m: jnp.trace(m))(V_xx)

        PDE_residual = (
            V_t.reshape(-1)
            + jnp.einsum("bd,bd->b", bbar, V_x)
            + 0.5 * disp2 * trace_term
        ).reshape(-1, 1)
        L1 = jnp.mean(jnp.square(PDE_residual))

        # Terminal condition loss (preserved from existing code)
        V_term = self.net.apply(params, t_terminal, x_terminal)
        diff = x_terminal @ self.C.T - jnp.array([0.0111358, 0.56246203])
        target = jnp.exp(-jnp.linalg.norm(diff, axis=1, ord=2)).reshape(-1, 1)
        L3 = jnp.mean(jnp.square(V_term - target))

        return L1, L3, (L1 + L3)
