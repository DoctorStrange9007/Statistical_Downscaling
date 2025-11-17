import jax
import jax.numpy as jnp
import jax.lax as lax
from jax.tree_util import tree_map
from src.optimal_transport.simple_NN import make_mlp
from functools import partial


class PolicyGradient:
    """
    Implements a linear Markov Decision Process (MDP) for statistical downscaling.

    This class provides methods for optimizing mixture model parameters using gradient descent
    to minimize the KL divergence between a parameterized model and a target transition kernel.
    The optimization includes both cost terms and KL divergence regularization.

    The model operates on pairs of states (y, y') and can use different forms parameterized model and target kernel
    (e.g. mixture of Gaussians).
    """

    def __init__(self, run_sett: dict, normalized_flow_model, true_data_model):
        """
        Initialize the PolicyGradient object.
        """
        self.run_sett = run_sett  # Store the full run_sett
        self.beta = self.run_sett["beta"]
        self.K = self.run_sett["K"]
        self.N = self.run_sett["N"]
        self.d_prime = self.run_sett["d_prime"]
        self.true_data_model = true_data_model
        self.normalized_flow_model = normalized_flow_model
        self.B = self.run_sett["B"]
        lr_sett = self.run_sett["lrates"]
        self.lrate = float(
            lr_sett[0] if isinstance(lr_sett, (list, tuple)) else lr_sett
        )

    def per_trajectory_joint_score(self, n, y_traj, y_traj_prime):
        """
        Compute gradient of q_n density w.r.t. params at index n, for a single trajectory.
        Uses JAX control flow to support tracer n from vmap.
        """

        def make_branch(k: int):
            if k == 0:

                def branch(_op=None, k=k):
                    y0 = y_traj[0].reshape(-1)
                    y0p = y_traj_prime[0].reshape(-1)

                    def mlp_scalar(params0):
                        raw = self.normalized_flow_model._apply_0(
                            params0, jnp.zeros((2 * self.d_prime,))
                        )
                        return jnp.sum(raw)

                    return jax.grad(mlp_scalar)(
                        self.normalized_flow_model.param_models[0]
                    )

                return branch
            else:

                def branch(_op=None, k=k):
                    prev = jnp.concatenate(
                        [y_traj[k - 1].reshape(-1), y_traj_prime[k - 1].reshape(-1)],
                        axis=0,
                    )

                    def mlp_scalar(params_k):
                        raw = self.normalized_flow_model._apply_cond(params_k, prev)
                        return jnp.sum(raw)

                    return jax.grad(mlp_scalar)(
                        self.normalized_flow_model.param_models[k]
                    )

                return branch

        branches = [make_branch(k) for k in range(self.N + 1)]
        return lax.switch(n, branches, None)

    def per_trajectory_joint_cost(self, n, y_traj, y_traj_prime):
        """
        Compute the joint cost for a trajectory.
        """
        y_n_to_N = y_traj[n:]
        y_n_prime_to_N = y_traj_prime[n:]
        return (1 / 2) * jnp.sum((y_n_to_N - y_n_prime_to_N) ** 2)

    @partial(jax.jit, static_argnums=0)
    def G_val_run_per_trajectory(self, key):
        y_traj, y_traj_prime = self.normalized_flow_model.sample_trajectory(key)

        # Initialize zero-like gradients for each model (0..N)
        grads_all = [
            tree_map(jnp.zeros_like, p) for p in self.normalized_flow_model.param_models
        ]

        # Accumulate per-step gradients scaled by per-step cost
        for k in range(self.N + 1):
            grad_k = self.per_trajectory_joint_score(k, y_traj, y_traj_prime)
            cost_k = self.per_trajectory_joint_cost(k, y_traj, y_traj_prime)
            scaled_grad_k = tree_map(lambda g: g * cost_k, grad_k)
            grads_all[k] = tree_map(lambda a, b: a + b, grads_all[k], scaled_grad_k)

        # Return as a tuple of pytrees so vmap can handle it
        return tuple(grads_all)

    def G_val(self, key):
        # Map over trajectories and sum across batch on each parameter tree
        keys = jax.random.split(key, self.B)
        result = jax.vmap(self.G_val_run_per_trajectory, in_axes=(0,))(keys)
        grads_sum = [
            tree_map(lambda x: x.sum(axis=0), result[i]) for i in range(self.N + 1)
        ]
        return grads_sum

    @partial(jax.jit, static_argnums=0)
    def G_KL_run_per_trajectory(self, key):
        z_traj, z_traj_prime = self.true_data_model.sample_true_trajectory(key)
        grads_all = [
            tree_map(jnp.zeros_like, p) for p in self.normalized_flow_model.param_models
        ]
        for k in range(self.N + 1):
            grad_k = self.per_trajectory_joint_score(k, z_traj, z_traj_prime)
            grads_all[k] = tree_map(lambda a, b: a + b, grads_all[k], grad_k)

        return tuple(grads_all)

    def G_KL(self, key):

        keys = jax.random.split(key, self.B)
        result = jax.vmap(self.G_KL_run_per_trajectory, in_axes=(0,))(keys)
        grads_sum = [
            tree_map(lambda x: -2 * x.sum(axis=0), result[i]) for i in range(self.N + 1)
        ]
        return grads_sum

    def aggregate_g(self, key):
        key_model, key_true = jax.random.split(key)
        g_val = self.G_val(key_model)
        g_kl = self.G_KL(key_true)
        # Scale by batch size and beta
        return [
            tree_map(
                lambda gv, gk: (1 / self.B) * gv + (self.beta / self.B) * gk, gv, gk
            )
            for gv, gk in zip(g_val, g_kl)
        ]

    def update_params(self, key):

        updated_list = []
        for params_tree, aggregate_g_n in zip(
            self.normalized_flow_model.param_models, self.aggregate_g(key)
        ):
            updated_params = tree_map(
                lambda p, g: p - self.lrate * g, params_tree, aggregate_g_n
            )
            updated_list.append(updated_params)
        self.normalized_flow_model.param_models = updated_list


class NormalizedFlowModel:
    """
    Implements a normalized flow model for statistical downscaling.
    """

    def __init__(self, run_sett: dict):
        """
        Initialize the NormalizedFlowModel object.
        """
        self.run_sett = run_sett
        self.N = self.run_sett["N"]
        self.d_prime = self.run_sett["d_prime"]

        # No internal PRNG; keys are passed to pure sampling functions

        # MLP config
        hidden_dims = tuple(self.run_sett["hidden_dims"])  # (64, 64)
        activation = self.run_sett["activation"]

        # Output parameterization size:
        # For vector [y, y'] of size 2*d', predict mean(2*d') and log_std(2*d') ⇒ 4*d'
        out_dim_params = 4 * self.d_prime

        # q_0 network: unconditional, take a constant zero input of size 1
        # Initialize params with a public seed for determinism; caller controls sampling keys
        self.param_0, self._apply_0 = make_mlp(
            jax.random.PRNGKey(self.run_sett["seed"]),
            in_dim=2 * self.d_prime,
            out_dim=out_dim_params,
            hidden_dims=hidden_dims,
            activation=activation,
            final_activation="identity",
        )

        # q_n networks for n=1..N: conditional on (y_{n-1}, y'_{n-1}) of size 2*d'
        self.param_n = []
        self._apply_cond = None
        for n in range(1, self.N + 1):
            kn = jax.random.fold_in(jax.random.PRNGKey(self.run_sett["seed"]), n)
            params_n, apply_fn = make_mlp(
                kn,
                in_dim=2 * self.d_prime,
                out_dim=out_dim_params,
                hidden_dims=hidden_dims,
                activation=activation,
                final_activation="identity",
            )
            self.param_n.append(params_n)
            self._apply_cond = apply_fn  # identical shapes; reuse reference

        # Expose list of all parameters (index 0 for q_0, index n for q_n)
        self.param_models = [self.param_0] + self.param_n

    @partial(jax.jit, static_argnums=0)
    def sample_trajectory(self, key):
        """
        Sample an entire trajectory (y_0..y_N, y'_0..y'_N) using a pure, key-driven path.
        Shapes:
          - returns y_traj, y_traj_prime with shape (N+1, d_prime, 1)
        """
        d_prime = self.d_prime
        # Initial (unconditional)
        key, k0 = jax.random.split(key)
        raw0 = self._apply_0(self.param_0, jnp.zeros((2 * d_prime,)))
        mean0, log_std0 = self._split_mean_logstd(raw0, d_prime=d_prime)
        joint0 = self._sample_diag_gauss(k0, mean0.reshape(-1), log_std0.reshape(-1))
        y0 = joint0[:d_prime].reshape((d_prime, 1))
        y0p = joint0[d_prime:].reshape((d_prime, 1))
        # Stack per-step parameter trees across steps so scan can slice along time axis
        # param_n_stacked leaves will have shape (N, ..leaf_shape..)
        param_n_stacked = tree_map(
            lambda *elems: jnp.stack(elems, axis=0), *self.param_n
        )

        # Scan over conditional steps with params provided per step
        def step(carry, params_k):
            prev_y, prev_yp, key_in = carry
            key_in, ks = jax.random.split(key_in)
            x_prev = jnp.concatenate([prev_y.reshape(-1), prev_yp.reshape(-1)], axis=0)
            raw_k = self._apply_cond(params_k, x_prev)
            mean_k, log_std_k = self._split_mean_logstd(raw_k, d_prime=d_prime)
            joint_k = self._sample_diag_gauss(
                ks, mean_k.reshape(-1), log_std_k.reshape(-1)
            )
            yk = joint_k[:d_prime].reshape((d_prime, 1))
            ypk = joint_k[d_prime:].reshape((d_prime, 1))
            return (yk, ypk, key_in), (yk, ypk)

        init_carry = (y0, y0p, key)
        (_, _, _), (ys, yps) = lax.scan(step, init_carry, xs=param_n_stacked)
        # Prepend initial state
        y_traj = jnp.concatenate([y0[None, ...], ys], axis=0)
        y_traj_prime = jnp.concatenate([y0p[None, ...], yps], axis=0)
        return y_traj, y_traj_prime

    @staticmethod
    def _split_mean_logstd(raw: jax.Array, d_prime: int) -> tuple[jax.Array, jax.Array]:
        # raw: (4*d') → mean(2*d'), log_std(2*d') and clamp log_std
        two_d = 2 * d_prime
        mean = raw[..., :two_d]
        log_std = raw[..., two_d : 2 * two_d]
        log_std = jnp.clip(log_std, a_min=-5.0, a_max=2.0)
        return mean, log_std

    @staticmethod
    def _logpdf_diag_gauss(
        x: jax.Array, mean: jax.Array, log_std: jax.Array
    ) -> jax.Array:
        var = jnp.exp(2.0 * log_std)
        log2pi = jnp.log(2.0 * jnp.pi)
        return -0.5 * (jnp.sum(log2pi + 2.0 * log_std + (x - mean) ** 2 / var))

    @staticmethod
    def _sample_diag_gauss(
        key: jax.Array, mean: jax.Array, log_std: jax.Array
    ) -> jax.Array:
        eps = jax.random.normal(key, shape=mean.shape)
        return mean + jnp.exp(log_std) * eps

    def q_0(self, y0, y0_prime):
        """
        q_0 modeled as a diagonal Gaussian parameterized by an MLP with constant input.
        - If y0, y0_prime are provided: return density q_0(y0, y0_prime).
        - Else: sample (y0, y0_prime).
        """
        d_prime = self.d_prime
        raw = self._apply_0(self.param_0, jnp.zeros((2 * d_prime,)))
        mean, log_std = self._split_mean_logstd(raw, d_prime=d_prime)

        if y0 is not None and y0_prime is not None:
            joint = jnp.concatenate([y0.reshape(-1), y0_prime.reshape(-1)], axis=0)
            logp = self._logpdf_diag_gauss(joint, mean.reshape(-1), log_std.reshape(-1))
            return jnp.exp(logp)

    def q_n(self, n, ynm1, ynm1_prime, yn, yn_prime):
        """
        q_n modeled as a diagonal Gaussian with parameters given by MLP([y_{n-1}, y'_{n-1}]).
        - If yn, yn_prime are provided: return density q_n(yn, yn_prime | ynm1, ynm1_prime).
        - Else: sample (yn, yn_prime).
        """
        assert 1 <= n <= self.N, "n must be in 1..N"
        d_prime = self.d_prime
        x_prev = jnp.concatenate(
            [ynm1.reshape(-1), ynm1_prime.reshape(-1)], axis=0
        )  # (2*d',)
        raw = self._apply_cond(self.param_n[n - 1], x_prev)
        mean, log_std = self._split_mean_logstd(raw, d_prime=d_prime)

        if yn is not None and yn_prime is not None:
            joint = jnp.concatenate([yn.reshape(-1), yn_prime.reshape(-1)], axis=0)
            logp = self._logpdf_diag_gauss(joint, mean.reshape(-1), log_std.reshape(-1))
            return jnp.exp(logp)


class TrueDataModel:
    """
    Implements a true data model for statistical downscaling.
    """

    def __init__(self, run_sett: dict):
        """
        Initialize the TrueDataModel object.
        """
        self.run_sett = run_sett
        self.N = self.run_sett["N"]
        self.d_prime = self.run_sett["d_prime"]

    @partial(jax.jit, static_argnums=0)
    def sample_true_trajectory(self, key):
        """
        Pure, key-driven sampler for the true trajectory (JAX-friendly).
        Returns (y_traj, y_traj_prime) with shape (N+1, d_prime, 1).
        """
        d = self.d_prime
        N = self.N
        # Fixed GMM params
        means_y = jnp.stack([1.0 * jnp.ones((d,)), -1.0 * jnp.ones((d,))])  # (2, d)
        stds_y = jnp.stack([0.6 * jnp.ones((d,)), 0.8 * jnp.ones((d,))])  # (2, d)
        weights_y = jnp.array([0.5, 0.5])
        means_yprime = jnp.stack(
            [1.5 * jnp.ones((d,)), -1.5 * jnp.ones((d,))]
        )  # (2, d)
        stds_yprime = jnp.stack([0.5 * jnp.ones((d,)), 0.7 * jnp.ones((d,))])  # (2, d)
        weights_yprime = jnp.array([0.6, 0.4])
        # Keys
        key_y, key_yprime = jax.random.split(key)
        keys_y = jax.random.split(key_y, N + 1)
        keys_yprime = jax.random.split(key_yprime, N + 1)

        # Vmappable GMM sampler
        def _sample_gmm_key(k, means, stds, weights):
            k1, k2 = jax.random.split(k)
            comp = jax.random.categorical(k1, jnp.log(weights))
            mean = means[comp]  # (d,)
            std = stds[comp]  # (d,)
            z = jax.random.normal(k2, shape=(d,))
            return (mean + std * z).reshape((d, 1))

        y_traj = jax.vmap(_sample_gmm_key, in_axes=(0, None, None, None))(
            keys_y, means_y, stds_y, weights_y
        )
        y_traj_prime = jax.vmap(_sample_gmm_key, in_axes=(0, None, None, None))(
            keys_yprime, means_yprime, stds_yprime, weights_yprime
        )
        return y_traj, y_traj_prime
