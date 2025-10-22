"""Base PDE solver utilities built on JAX.

Defines `PDE_solver`, a base class providing network setup, sampling utilities,
training loop scaffolding, and gradient helpers. Concrete solvers should
subclass it and implement `loss_fn`.
"""

import jax
import jax.numpy as jnp
import optax
from functools import partial
from src.generation.DGMJax import DGMNetJax
import orbax.checkpoint as ocp
import os
import shutil


class PDE_solver:
    """Base class for PDE solvers.

    Provides common initialization, sampling, forward pass, training loop,
    and gradient utilities. Subclasses must implement `loss_fn`.
    """

    def __init__(self, settings: dict, rng_key: jax.Array | None = None):
        """Initialize solver configuration, model parameters, and optimizer.

        Args:
            settings: Dict with sections `general`, `pde_solver`, and `DGM`
                specifying domain bounds, sampling sizes, optimizer
                hyperparameters, and network architecture.
            rng_key: Optional PRNG key for deterministic init and sampling.
        """
        # Common settings
        self.t_low = float(settings["pde_solver"]["t_low"])
        self.T = float(settings["general"]["T"])
        self.x_low = float(settings["pde_solver"]["x_low"])
        self.x_high = float(settings["pde_solver"]["x_high"])
        self.nSim_interior = int(settings["pde_solver"]["nSim_interior"])
        self.nSim_terminal = int(settings["pde_solver"]["nSim_terminal"])
        self.sampling_stages = int(settings["pde_solver"]["sampling_stages"])
        self.steps_per_sample = int(settings["pde_solver"]["steps_per_sample"])
        self.d = int(settings["general"]["d"])

        # Network
        hidden = int(settings["DGM"]["nodes_per_layer"])
        n_layers = int(settings["DGM"]["num_layers"])
        self.num_models = int(settings["pde_solver"]["num_models"])
        self.net = DGMNetJax(
            input_dim=self.d, layer_width=hidden, num_layers=n_layers, final_trans=None
        )

        self.rng = rng_key if rng_key is not None else jax.random.PRNGKey(0)
        # Initialize parameters with dummy batch
        t0 = jnp.zeros((1, 1), dtype=jnp.float32)
        x0 = jnp.zeros((1, self.d), dtype=jnp.float32)
        keys = jax.random.split(self.rng, max(2, self.num_models + 1))
        self.rng = keys[0]
        self.params_list = [
            self.net.init(keys[k + 1], t0, x0) for k in range(self.num_models)
        ]

        # Optimizer
        if settings["pde_solver"]["learning_rate"] == "sirignano":
            boundaries = settings["pde_solver"]["boundaries"]
            schedules = [
                optax.constant_schedule(1e-4),
                optax.constant_schedule(5e-4),
                optax.constant_schedule(1e-5),
                optax.constant_schedule(5e-6),
                optax.constant_schedule(1e-6),
                optax.constant_schedule(5e-7),
                optax.constant_schedule(1e-7),
            ]
            self.learning_rate = optax.join_schedules(
                schedules=schedules, boundaries=boundaries
            )
        else:
            self.learning_rate = float(settings["pde_solver"]["learning_rate"])
        self.optimizer = optax.adam(self.learning_rate)
        # Optimizer state per model
        self.opt_state_list = [self.optimizer.init(p) for p in self.params_list]

    # Sampling utilities (use JAX RNG)
    def sampler(self):
        """Sample interior and terminal training points for the PDE.

        Returns:
            Tuple `(t_interior, x_interior, t_terminal, x_terminal)` where
            interior samples are uniform in time over `(t_low, T]` and uniform
            in space over `[x_low, x_high]`, and terminal times are fixed at `T`.
        """
        self.rng, k1, k2, k3 = jax.random.split(self.rng, 4)
        t_interior = jax.random.uniform(
            k1, shape=(self.nSim_interior, 1), minval=self.t_low, maxval=self.T
        )
        x_interior = jax.random.uniform(
            k2,
            shape=(self.nSim_interior, self.d),
            minval=self.x_low,
            maxval=self.x_high,
        )
        t_terminal = jnp.ones((self.nSim_terminal, 1), dtype=jnp.float32) * self.T
        x_terminal = jax.random.uniform(
            k3,
            shape=(self.nSim_terminal, self.d),
            minval=self.x_low,
            maxval=self.x_high,
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

        Iterates for `sampling_stages`, drawing fresh samples each stage and
        performing `steps_per_sample` optimization steps per model.

        Args:
            log_fn: Optional callable receiving a dict of scalar metrics once
                per sampling stage.
        """
        for i in range(self.sampling_stages):
            t_interior, x_interior, t_terminal, x_terminal = self.sampler()
            for _ in range(self.steps_per_sample):
                for k in range(self.num_models):

                    def total_loss_k(p, idx=k):
                        y_arr = self.y_bar
                        # Allow shapes: (M, out_dim), (M, out_dim, 1), or (out_dim,)
                        y_t = y_arr[idx] if getattr(y_arr, "ndim", 0) >= 2 else y_arr
                        return self.loss_fn(
                            p,
                            t_interior,
                            x_interior,
                            t_terminal,
                            x_terminal,
                            y_bar=y_t,
                        )[2]

                    _, grads = jax.value_and_grad(total_loss_k)(self.params_list[k])
                    updates, self.opt_state_list[k] = self.optimizer.update(
                        grads, self.opt_state_list[k]
                    )
                    self.params_list[k] = optax.apply_updates(
                        self.params_list[k], updates
                    )

            for k in range(self.num_models):
                y_arr = self.y_bar
                y_t = y_arr[k] if getattr(y_arr, "ndim", 0) >= 2 else y_arr
                L1, L3, tot = self.loss_fn(
                    self.params_list[k],
                    t_interior,
                    x_interior,
                    t_terminal,
                    x_terminal,
                    y_bar=y_t,
                )
                print(
                    f"Stage {i} [model {k}]: Loss = {float(tot):.6f}, L1 = {float(L1):.6f}, L3 = {float(L3):.6f}"
                )
                log_fn(
                    {
                        "step": int(i),
                        f"pde_solver/model_{k}/loss_total": float(tot),
                        f"pde_solver/model_{k}/loss_PDE": float(L1),
                        f"pde_solver/model_{k}/loss_terminal": float(L3),
                    }
                )

    @partial(jax.jit, static_argnums=0)
    def grad_log_h_params(
        self, params: jax.Array, x: jax.Array, t: jax.Array
    ) -> jax.Array:
        """Compute per-sample gradients ∂/∂x log h(t, x) for given params.

        Returns an array matching the flattened spatial dimension of `x`.
        """
        x_flat = x.reshape(1, -1)
        t = t.reshape(1, 1)

        def loss(xs):
            h = self.net.apply(params, t, xs).squeeze(-1)
            return jnp.sum(jnp.log(jnp.maximum(h, 1e-6)))

        return jax.grad(loss)(x_flat)

    def grad_log_h_model(self, model_idx: int, x: jax.Array, t: jax.Array) -> jax.Array:
        """Convenience wrapper to compute `grad_log_h` for model `model_idx`."""
        return self.grad_log_h_params(self.params_list[model_idx], x, t)

    def grad_log_h_all(self, x: jax.Array, t: jax.Array):
        """Compute `grad_log_h` for all models and return a Python list of arrays."""
        return [self.grad_log_h_params(p, x, t) for p in self.params_list]

    # Backward-compatible single-model API
    @partial(jax.jit, static_argnums=0)
    def grad_log_h(self, x: jax.Array, t: jax.Array) -> jax.Array:
        """Backward-compatible wrapper that uses model 0 parameters."""
        return self.grad_log_h_params(self.params_list[0], x, t)

    # JAX-friendly per-model selection for num_samples == num_models
    def _grad_log_h_params_switch(
        self, idx: jax.Array, x: jax.Array, t: jax.Array
    ) -> jax.Array:
        """Select params for model `idx` using a JAX switch and compute `grad_log_h`.

        Avoids traced indexing into Python lists inside jit/vmap.
        """

        def make_case(k):
            return lambda x_=x, t_=t, k_=k: self.grad_log_h_params(
                self.params_list[k_], x_, t_
            )

        return jax.lax.switch(idx, tuple(make_case(k) for k in range(self.num_models)))

    @partial(jax.jit, static_argnums=0)
    def grad_log_h_batched_one_per_model(self, x: jax.Array, t: jax.Array) -> jax.Array:
        """Compute grad_log_h for a batch where sample i uses model i.

        Args:
            x: Array with shape `(M, d)` or `(M, d, 1)` where `M == self.num_models`.
            t: Scalar `()` or array with shape `(M,)` or `(M, 1)`.

        Returns:
            Array with shape `(M, d)` containing per-sample gradients.
        """
        M = x.shape[0]
        x_flat = x.reshape(M, -1)
        if t.ndim == 0:
            t = jnp.ones((M, 1), dtype=x_flat.dtype) * t
        elif t.ndim == 1:
            t = t.reshape(M, 1)
        idxs = jnp.arange(M, dtype=jnp.int32)

        def per_sample(i, xs, ts):
            return self._grad_log_h_params_switch(i, xs, ts)

        grads = jax.vmap(per_sample)(idxs, x_flat, t)
        # grads has shape (M, 1, d_flat); squeeze the singleton batch axis
        if grads.ndim == 3 and grads.shape[1] == 1:
            grads = jnp.squeeze(grads, axis=1)
        return grads

    def save_params(self, ckpt_dir: str):
        """Save current params_list and opt_state_list to a checkpoint directory.

        Args:
            ckpt_dir: Directory path to write the checkpoint. This directory will be
                created if it doesn't exist, and its contents will be overwritten.
        """
        # Ensure the target directory exists and is empty for a clean save.
        if os.path.exists(ckpt_dir):
            shutil.rmtree(ckpt_dir)
        checkpointer = ocp.PyTreeCheckpointer()
        # Create a payload containing both the parameters and the optimizer state.
        payload = {
            "params_list": self.params_list,
            "opt_state_list": self.opt_state_list,
        }
        # Save the complete state directly into the specified directory.
        checkpointer.save(ckpt_dir, payload)
        print(f"Solver state saved successfully to: {ckpt_dir}")

    def load_params(self, ckpt_dir: str) -> bool:
        """Load params_list and opt_state_list from a checkpoint directory.

        Returns True if loaded successfully, else False.
        """
        try:
            if not hasattr(self, "params_list") or not hasattr(self, "opt_state_list"):
                print("Error: Solver must be initialized before loading checkpoint.")
                print("       (self.params_list and self.opt_state_list must exist)")
                return False

            target_item = {
                "params_list": self.params_list,
                "opt_state_list": self.opt_state_list,
            }

            checkpointer = ocp.PyTreeCheckpointer()
            restored = checkpointer.restore(ckpt_dir, item=target_item)

            if restored:
                self.params_list = restored["params_list"]
                self.opt_state_list = restored["opt_state_list"]
                print(f"Solver state loaded successfully from: {ckpt_dir}")
                return True
            else:
                print(f"Failed to find or restore checkpoint from: {ckpt_dir}")
                return False

        except Exception as e:
            print(f"Failed to load solver state from {ckpt_dir}. Error: {e}")
            return False

    def load_params_old(self, ckpt_dir: str) -> bool:
        """CAN DELETE LATER BUT CURRENT SAMPLE RESULTS BASED ON THIS VERSION"""
        try:
            checkpointer = ocp.PyTreeCheckpointer()
            restored = checkpointer.restore(ckpt_dir)

            if restored and "params_list" in restored and "opt_state_list" in restored:
                # Load both the parameters and the optimizer state.
                self.params_list = restored["params_list"]
                self.opt_state_list = restored["opt_state_list"]
                print(f"Solver state loaded successfully from: {ckpt_dir}")
                return True
            else:
                print(f"Checkpoint in {ckpt_dir} is missing required data.")
                return False

        except Exception as e:
            print(f"Failed to load solver state from {ckpt_dir}. Error: {e}")
            return False
