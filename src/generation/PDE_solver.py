import jax
import jax.numpy as jnp
import numpy as np
import optax
from DGMJax import DGMNetJax
from functools import partial


# DGMNetJax, LSTMLayerJax, DenseLayerJax moved to DGMJax.py


class PDE_solver:
    def __init__(self, grad_log, samples, settings, rng_key: jax.Array | None = None):
        # Settings
        self.t_low = float(settings["pde_solver"]["t_low"])
        self.T = float(settings["general"]["T"])
        self.x_low = float(settings["pde_solver"]["x_low"])
        self.nSim_interior = int(settings["pde_solver"]["nSim_interior"])
        self.nSim_terminal = int(settings["pde_solver"]["nSim_terminal"])
        self.x_high = float(settings["pde_solver"]["x_high"])
        self.x_multiplier = float(settings["pde_solver"]["x_multiplier"])
        self.beta_min = float(settings["general"]["beta_min"])
        self.beta_max = float(settings["general"]["beta_max"])
        self.d = int(settings["general"]["d"])
        self.sampling_stages = int(settings["pde_solver"]["sampling_stages"])
        self.steps_per_sample = int(settings["pde_solver"]["steps_per_sample"])
        self.learning_rate = float(settings["pde_solver"]["learning_rate"])

        # Linear constraint Cx = y
        self.C = jnp.array(settings["pde_solver"]["C"], dtype=jnp.float32)
        self.y = jnp.array(
            samples @ np.array(settings["pde_solver"]["C"]).T, dtype=jnp.float32
        )

        # JAX model (Flax implementation of DGM architecture)
        hidden = (
            int(settings["pre_trained"]["model"]["nodes_per_layer"])
            if "pre_trained" in settings
            else 128
        )
        n_layers = (
            int(settings["pre_trained"]["model"]["num_layers"])
            if "pre_trained" in settings
            else 3
        )
        self.net = DGMNetJax(
            input_dim=self.d, layer_width=hidden, num_layers=n_layers, final_trans=None
        )
        self.rng = rng_key if rng_key is not None else jax.random.PRNGKey(0)
        # Initialize parameters with dummy batch of size 1
        t0 = jnp.zeros((1, 1), dtype=jnp.float32)
        x0 = jnp.zeros((1, self.d), dtype=jnp.float32)
        self.params = self.net.init(self.rng, t0, x0)

        # Optimizer
        self.optimizer = optax.adam(self.learning_rate)
        self.opt_state = self.optimizer.init(self.params)

        # JAX-based score function from pre_trained_model
        self.grad_log = grad_log  # Callable: (x, t) -> grad log p

    # Drift/diffusion schedule
    def beta(self, t: jax.Array) -> jax.Array:
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def f(self, x: jax.Array, t: jax.Array) -> jax.Array:
        return -0.5 * self.beta(t) * x

    def g(self, t: jax.Array) -> jax.Array:
        return jnp.sqrt(self.beta(t))

    def b_bar(self, t: jax.Array, x: jax.Array) -> jax.Array:
        disp = self.g(self.T - t)
        return -self.f(x, self.T - t) + self.grad_log(x, self.T - t) * (disp**2)

    # Sampling utilities (use JAX RNG)
    def sampler(self):
        self.rng, k1, k2, k3 = jax.random.split(self.rng, 4)
        t_interior = jax.random.uniform(
            k1, shape=(self.nSim_interior, 1), minval=self.t_low, maxval=self.T
        )
        x_interior = jax.random.uniform(
            k2,
            shape=(self.nSim_interior, self.d),
            minval=self.x_low,
            maxval=self.x_high * self.x_multiplier,
        )
        t_terminal = jnp.ones((self.nSim_terminal, 1), dtype=jnp.float32) * self.T
        x_terminal = jax.random.uniform(
            k3,
            shape=(self.nSim_terminal, self.d),
            minval=self.x_low,
            maxval=self.x_high * self.x_multiplier,
        )
        return t_interior, x_interior, t_terminal, x_terminal

    # Value function V(t, x)
    @partial(jax.jit, static_argnums=0)
    def V(self, params, t: jax.Array, x: jax.Array) -> jax.Array:
        return self.net.apply(params, t, x)

    # Loss computation (no jit first for simplicity)
    @partial(jax.jit, static_argnums=0)
    def loss_fn(self, params, t_interior, x_interior, t_terminal, x_terminal):
        # Per-sample scalar V for gradient/hessian computations
        def V_single(ts: jax.Array, xs: jax.Array) -> jax.Array:
            ts_b = ts.reshape(1, 1)
            xs_b = xs.reshape(1, -1)
            return self.net.apply(params, ts_b, xs_b).squeeze()

        # Gradients and Hessians w.r.t. t and x
        dV_dt_fn = jax.grad(lambda ts, xs: V_single(ts, xs), argnums=0)
        dV_dx_fn = jax.grad(lambda xs, ts: V_single(ts, xs))  # grad w.r.t. x
        H_fn = jax.hessian(lambda xs, ts: V_single(ts, xs))

        # Vectorize over batch
        V_t = jax.vmap(lambda ts, xs: dV_dt_fn(ts.squeeze(), xs))(
            t_interior, x_interior
        )
        V_x = jax.vmap(lambda ts, xs: dV_dx_fn(xs, ts.squeeze()))(
            t_interior, x_interior
        )
        V_xx = jax.vmap(lambda ts, xs: H_fn(xs, ts.squeeze()))(t_interior, x_interior)

        # b_bar and diffusion trace term
        bbar = self.b_bar(t_interior, x_interior)  # (batch, d)
        disp2 = (self.g(self.T - t_interior) ** 2).reshape(-1)
        trace_term = jax.vmap(lambda m: jnp.trace(m))(V_xx)  # (batch,)

        PDE_residual = (
            V_t.reshape(-1)
            + jnp.einsum("bd,bd->b", bbar, V_x)
            + 0.5 * disp2 * trace_term
        ).reshape(-1, 1)

        L1 = jnp.mean(jnp.square(PDE_residual))

        # Terminal condition loss
        V_term = self.net.apply(params, t_terminal, x_terminal)  # (batch_T, 1)
        diff = x_terminal @ self.C.T - self.y  # broadcast y over rows
        target = jnp.exp(-jnp.linalg.norm(diff, axis=1, ord=2)).reshape(-1, 1)
        L3 = jnp.mean(jnp.square(V_term - target))

        return L1, L3, (L1 + L3)

    def train(self):
        for i in range(self.sampling_stages):
            t_interior, x_interior, t_terminal, x_terminal = self.sampler()

            for _ in range(self.steps_per_sample):
                # Compute loss and grads
                def total_loss(p):
                    _, _, tot = self.loss_fn(
                        p, t_interior, x_interior, t_terminal, x_terminal
                    )
                    return tot

                loss_val, grads = jax.value_and_grad(total_loss)(self.params)
                updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
                self.params = optax.apply_updates(self.params, updates)

            L1, L3, tot = self.loss_fn(
                self.params, t_interior, x_interior, t_terminal, x_terminal
            )
            print(
                f"Stage {i}: Loss = {float(tot):.6f}, L1 = {float(L1):.6f}, L3 = {float(L3):.6f}"
            )

    @partial(jax.jit, static_argnums=0)
    def grad_log_h(self, x: jax.Array, t: jax.Array) -> jax.Array:
        # Compute grad wrt x of log h(t, x)
        def log_h_single(ts: jax.Array, xs: jax.Array) -> jax.Array:
            ts_b = ts.reshape(1, 1)
            xs_b = xs.reshape(1, -1)
            h = self.net.apply(self.params, ts_b, xs_b).squeeze()
            return jnp.log(jnp.maximum(h, 1e-6))

        grad_fn = jax.grad(lambda xs, ts: log_h_single(ts, xs))
        grads = jax.vmap(lambda ts, xs: grad_fn(xs, ts.squeeze()))(t, x)
        return grads
