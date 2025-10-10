import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from src.generation.PDE_solver import PDE_solver
from swirl_dynamics.lib.diffusion import diffusion
from typing import Any, TypeAlias
from collections.abc import Mapping


Array: TypeAlias = jax.Array
ArrayMapping: TypeAlias = Mapping[str, Array]
Params: TypeAlias = Mapping[str, Any]


def dlog_dt(f: diffusion.ScheduleFn) -> diffusion.ScheduleFn:
    """Returns d/dt log(f(t)) = ḟ(t)/f(t) given f(t)."""
    return jax.grad(lambda t: jnp.log(f(t)))


def dsquare_dt(f: diffusion.ScheduleFn) -> diffusion.ScheduleFn:
    """Returns d/dt (f(t))^2 = 2ḟ(t)f(t) given f(t)."""
    return jax.grad(lambda t: jnp.square(f(t)))


class KSStatisticalDownscalingPDESolver(PDE_solver):
    """PDE for statistical downscaling with linear constraint Cx=y.

    Implements the loss used in the previous monolithic solver, including
    the reverse-time SDE drift components through grad_log and beta schedule.
    """

    def __init__(
        self,
        samples: np.ndarray,
        y_bar: np.ndarray,
        settings: dict,
        denoise_fn,
        scheme,
        rng_key: jax.Array | None = None,
    ):
        """Initialize the statistical downscaling PDE solver.

        Args:
            grad_log: Callable `(x, t) -> grad log p(x, t)` defining the score.
            samples: Training samples used for constructing constraints.
            settings: Hierarchical settings dictionary.
            rng_key: Optional PRNG key for deterministic initialization.
        """
        super().__init__(settings=settings, rng_key=rng_key)

        self.C_prime = jnp.array(
            [[1 if j == 4 * i else 0 for j in range(192)] for i in range(48)]
        )
        self.y_bar = jnp.asarray(y_bar)
        self.scheme = scheme
        self.denoise_fn = denoise_fn
        self.lambda_ = jnp.float32(settings["pde_solver"]["lambda"])

    @partial(jax.jit, static_argnums=0)
    def loss_fn(self, params, t_interior, x_interior, t_terminal, x_terminal, y_bar):
        """Compute PDE residual and terminal losses for downscaling.

        Returns:
            Tuple `(L1, L3, total)` where `L1` is interior residual MSE and
            `L3` enforces the terminal constraint via a smooth kernel.
        """

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
        # bbar = self.b_bar(t_interior, x_interior)
        # disp2 = (self.g(self.T - t_interior) ** 2).reshape(-1)
        trace_term = jax.vmap(lambda m: jnp.trace(m))(V_xx)

        def _drift(x: Array, t: Array) -> Array:
            x = x[
                None, ..., None
            ]  # do it sample for sample as we have to take s_n=(t_n, x_n) pair together
            assert not t.ndim, "`t` must be a scalar."
            s, sigma = self.scheme.scale(t), self.scheme.sigma(t)
            x_hat = jnp.divide(x, s)
            dlog_sigma_dt = dlog_dt(self.scheme.sigma)(t)
            dlog_s_dt = dlog_dt(self.scheme.scale)(t)
            drift = (2 * dlog_sigma_dt + dlog_s_dt) * x
            drift -= 2 * dlog_sigma_dt * s * self.denoise_fn(x_hat, sigma)

            return jnp.squeeze(drift)

        def _half_diffusion2(x: Array, t: Array) -> Array:
            assert not t.ndim, "`t` must be a scalar."
            s, sigma = self.scheme.scale(t), self.scheme.sigma(t)
            dsquare_sigma_dt = dsquare_dt(self.scheme.sigma)(t)
            return (1 / 2) * (s**2) * dsquare_sigma_dt

        drifts = jax.vmap(_drift)(x_interior, jnp.squeeze(t_interior, -1))
        diffusions = jax.vmap(_half_diffusion2)(x_interior, jnp.squeeze(t_interior, -1))

        PDE_residual = (
            V_t.reshape(-1)
            + jnp.einsum("bd,bd->b", drifts, V_x)
            + diffusions * trace_term
        ).reshape(-1, 1)
        L1 = jnp.mean(jnp.square(PDE_residual))

        # Terminal condition loss for a single y target (per model)
        V_term = self.net.apply(params, t_terminal, x_terminal)  # (B, 1)
        Cx = x_terminal @ self.C_prime.T  # (B, out_dim)

        # Use provided y_bar for this call; allow (out_dim,) or (out_dim,1)
        y_bar = jnp.squeeze(y_bar, axis=-1)
        diff = Cx - y_bar  # (B, out_dim)
        sqnorm = jnp.sum(jnp.square(diff), axis=-1)  # (B,)
        target = (1 / self.lambda_) * jnp.exp(-sqnorm / (self.lambda_**2))  # (B,)
        target = target.reshape(-1, 1)  # (B, 1)
        L3 = jnp.mean(jnp.square(V_term - target))

        return L1, L3, (L1 + L3)
