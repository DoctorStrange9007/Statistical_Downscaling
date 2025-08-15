import jax
import jax.numpy as jnp
import optax
from functools import partial

try:
    from .DGMJax import DGMNetJax
except ImportError:  # when run as a script from this folder
    from DGMJax import DGMNetJax


class PDE_solver:
    """Base class for PDE solvers.

    Provides common initialization, sampling, forward pass, training loop,
    and utility gradients. Subclasses must implement loss_fn.
    """

    def __init__(self, settings: dict, rng_key: jax.Array | None = None):
        """Initialize the base PDE solver with common configuration.

        Args:
            settings: Hierarchical settings containing keys under `general` and
                `pde_solver` specifying time horizon, sampling sizes, optimizer
                hyperparameters, spatial bounds, and network size.
            rng_key: Optional JAX PRNG key for deterministic initialization and
                sampling. If omitted, a default key is created.
        """
        # Common settings
        self.t_low = float(settings["pde_solver"]["t_low"])
        self.T = float(settings["general"]["T"])
        self.x_low = float(settings["pde_solver"]["x_low"])
        self.x_high = float(settings["pde_solver"]["x_high"])
        self.x_multiplier = float(settings["pde_solver"]["x_multiplier"])
        self.nSim_interior = int(settings["pde_solver"]["nSim_interior"])
        self.nSim_terminal = int(settings["pde_solver"]["nSim_terminal"])
        self.sampling_stages = int(settings["pde_solver"]["sampling_stages"])
        self.steps_per_sample = int(settings["pde_solver"]["steps_per_sample"])
        self.learning_rate = float(settings["pde_solver"]["learning_rate"])
        self.d = int(settings["general"]["d"])

        # Network
        hidden = int(
            settings.get("pre_trained", {}).get("model", {}).get("nodes_per_layer", 128)
        )
        n_layers = int(
            settings.get("pre_trained", {}).get("model", {}).get("num_layers", 3)
        )
        self.net = DGMNetJax(
            input_dim=self.d, layer_width=hidden, num_layers=n_layers, final_trans=None
        )

        self.rng = rng_key if rng_key is not None else jax.random.PRNGKey(0)
        # Initialize parameters with dummy batch
        t0 = jnp.zeros((1, 1), dtype=jnp.float32)
        x0 = jnp.zeros((1, self.d), dtype=jnp.float32)
        self.params = self.net.init(self.rng, t0, x0)

        # Optimizer
        self.optimizer = optax.adam(self.learning_rate)
        self.opt_state = self.optimizer.init(self.params)

    # Sampling utilities (use JAX RNG)
    def sampler(self):
        """Sample interior and terminal training points for the PDE.

        Returns:
            Tuple `(t_interior, x_interior, t_terminal, x_terminal)` where
            interior samples are uniform in time over `(t_low, T]` and uniform
            in space over `[x_low, x_high * x_multiplier]`, and terminal times
            are fixed at `T`.
        """
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
        """Evaluate the value network `V(t, x)`.

        Args:
            params: Network parameters.
            t: Time tensor of shape `(B, 1)`.
            x: Spatial tensor of shape `(B, d)`.

        Returns:
            Tensor of shape `(B, 1)` with the network output.
        """
        return self.net.apply(params, t, x)

    # Subclasses must implement this
    @partial(jax.jit, static_argnums=0)
    def loss_fn(self, params, t_interior, x_interior, t_terminal, x_terminal):
        """Compute loss terms. Must be implemented by subclasses.

        Returns:
            A tuple `(L1, L3, total)` of scalar losses.
        """
        raise NotImplementedError("loss_fn must be implemented by subclasses")

    def train(self, log_fn=None):
        """Train the network using SGD/Adam over sampled batches.

        Args:
            log_fn: Optional callable that accepts a dictionary of scalar
                metrics logged once per sampling stage.
        """
        for i in range(self.sampling_stages):
            t_interior, x_interior, t_terminal, x_terminal = self.sampler()

            for _ in range(self.steps_per_sample):

                def total_loss(p):
                    _, _, tot = self.loss_fn(
                        p, t_interior, x_interior, t_terminal, x_terminal
                    )
                    return tot

                _, grads = jax.value_and_grad(total_loss)(self.params)
                updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
                self.params = optax.apply_updates(self.params, updates)

            L1, L3, tot = self.loss_fn(
                self.params, t_interior, x_interior, t_terminal, x_terminal
            )
            print(
                f"Stage {i}: Loss = {float(tot):.6f}, L1 = {float(L1):.6f}, L3 = {float(L3):.6f}"
            )
            if log_fn is not None:
                try:
                    log_fn(
                        {
                            "stage": i,
                            "pde_solver/loss_total": float(tot),
                            "pde_solver/loss_PDE": float(L1),
                            "pde_solver/loss_terminal": float(L3),
                        }
                    )
                except Exception:
                    # Logging should not break training
                    pass

    @partial(jax.jit, static_argnums=0)
    def grad_log_h(self, x: jax.Array, t: jax.Array) -> jax.Array:
        """Compute gradients of log h(x, t) induced by the current network.

        Args:
            x: Spatial inputs of shape `(B, d)`.
            t: Time inputs of shape `(B, 1)`.

        Returns:
            Tensor of shape `(B, d)` with gradients w.r.t. x.
        """

        def log_h_single(ts: jax.Array, xs: jax.Array) -> jax.Array:
            ts_b = ts.reshape(1, 1)
            xs_b = xs.reshape(1, -1)
            h = self.net.apply(self.params, ts_b, xs_b).squeeze()
            return jnp.log(jnp.maximum(h, 1e-6))

        grad_fn = jax.grad(lambda xs, ts: log_h_single(ts, xs))
        grads = jax.vmap(lambda ts, xs: grad_fn(xs, ts.squeeze()))(t, x)
        return grads
