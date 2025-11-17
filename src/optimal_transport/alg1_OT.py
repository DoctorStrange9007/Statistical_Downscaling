import jax
import jax.numpy as jnp
import jax.lax as lax
from jax.tree_util import tree_map
from src.optimal_transport.simple_NN import make_mlp


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

    def get_trajectory(self):
        d_prime = self.d_prime
        N = self.N

        # Initialize trajectory arrays: shapes (N+1, d_prime, 1)
        y_traj = []
        y_traj_prime = []

        # Sample initial states
        y0, y0_prime = self.normalized_flow_model.q_0()
        y_traj.append(jnp.asarray(y0).reshape(d_prime, 1))
        y_traj_prime.append(jnp.asarray(y0_prime).reshape(d_prime, 1))

        # Sample transitions for n = 1, ..., N
        for n in range(1, N + 1):
            yn, yn_prime = self.normalized_flow_model.q_n(
                n, y_traj[n - 1], y_traj_prime[n - 1]
            )
            y_traj.append(jnp.asarray(yn).reshape(d_prime, 1))
            y_traj_prime.append(jnp.asarray(yn_prime).reshape(d_prime, 1))

        # Convert to arrays of shape (N+1, d_prime, 1)
        y_traj = jnp.stack(y_traj, axis=0)
        y_traj_prime = jnp.stack(y_traj_prime, axis=0)

        return y_traj, y_traj_prime

    def per_trajectory_joint_score(self, n, y_traj, y_traj_prime):
        """
        Compute gradient of q_n density w.r.t. params at index n, for a single trajectory.
        Uses JAX control flow to support tracer n from vmap.
        """
        nf = self.normalized_flow_model
        d = self.d_prime

        def make_branch(k: int):
            if k == 0:

                def branch(_op=None, k=k):
                    y0 = y_traj[0].reshape(-1)
                    y0p = y_traj_prime[0].reshape(-1)

                    def mlp_scalar(params0):
                        raw = nf._apply_0(params0, jnp.zeros((2 * d,)))
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
                        raw = nf._apply_cond(params_k, prev)
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

    def G_val_run_per_trajectory(self):
        y_traj, y_traj_prime = self.get_trajectory()

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

    def G_val(self):
        # Map over trajectories and sum across batch on each parameter tree
        result = jax.vmap(lambda _: self.G_val_run_per_trajectory())(jnp.arange(self.B))
        grads_sum = [
            tree_map(lambda x: x.sum(axis=0), result[i]) for i in range(self.N + 1)
        ]
        return grads_sum

    def G_KL_run_per_trajectory(self):
        z_traj, z_traj_prime = self.true_data_model.get_true_trajectory()
        grads_all = [
            tree_map(jnp.zeros_like, p) for p in self.normalized_flow_model.param_models
        ]
        for k in range(self.N + 1):
            grad_k = self.per_trajectory_joint_score(k, z_traj, z_traj_prime)
            grads_all[k] = tree_map(lambda a, b: a + b, grads_all[k], grad_k)

        return tuple(grads_all)

    def G_KL(self):

        result = jax.vmap(lambda _: self.G_KL_run_per_trajectory())(jnp.arange(self.B))
        grads_sum = [
            tree_map(lambda x: x.sum(axis=0), result[i]) for i in range(self.N + 1)
        ]
        return grads_sum

    def aggregate_g(self):
        g_val = self.G_val()
        g_kl = self.G_KL()
        # Scale by batch size and beta
        return [
            tree_map(
                lambda gv, gk: (1 / self.B) * gv + (self.beta / self.B) * gk, gv, gk
            )
            for gv, gk in zip(g_val, g_kl)
        ]

    def update_params(self):

        updated_list = []
        for params_tree, aggregate_g_n in zip(
            self.normalized_flow_model.param_models, self.aggregate_g()
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

        # PRNG state
        self._key = jax.random.PRNGKey(self.run_sett["seed"])

        # MLP config
        hidden_dims = tuple(self.run_sett["hidden_dims"])  # (64, 64)
        activation = self.run_sett["activation"]

        # Output parameterization size:
        # For vector [y, y'] of size 2*d', predict mean(2*d') and log_std(2*d') ⇒ 4*d'
        out_dim_params = 4 * self.d_prime

        # q_0 network: unconditional, take a constant zero input of size 1
        self._key, k0 = jax.random.split(self._key)
        self.param_0, self._apply_0 = make_mlp(
            k0,
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
            self._key, kn = jax.random.split(self._key)
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

    def q_0(self, y0=None, y0_prime=None):
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
        else:
            self._key, sk = jax.random.split(self._key)
            joint = self._sample_diag_gauss(sk, mean.reshape(-1), log_std.reshape(-1))
            y0 = joint[:d_prime].reshape((d_prime, 1))
            y0_prime = joint[d_prime:].reshape((d_prime, 1))
            return y0, y0_prime

    def q_n(self, n, ynm1, ynm1_prime, yn=None, yn_prime=None):
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
        else:
            self._key, sk = jax.random.split(self._key)
            joint = self._sample_diag_gauss(sk, mean.reshape(-1), log_std.reshape(-1))
            yn = joint[:d_prime].reshape((d_prime, 1))
            yn_prime = joint[d_prime:].reshape((d_prime, 1))
            return yn, yn_prime


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

    def get_true_trajectory(self):
        """
        Sample a true trajectory (y_0..y_N, y'_0..y'_N) from fixed GMMs.
        Returns:
            y_traj: Array (N+1, d_prime, 1)
            y_traj_prime: Array (N+1, d_prime, 1)
        """

        def _sample_gmm(key, d, means, stds, weights):
            # Sample from categorical to select component, then normal.
            k1, k2 = jax.random.split(key)
            comp = jax.random.categorical(k1, jnp.log(weights))
            mean = means[comp]  # (d,)
            std = stds[comp]  # (d,)
            z = jax.random.normal(k2, shape=(d,))
            return (mean + std * z).reshape((d, 1))

        d = self.d_prime
        N = self.N

        # Fixed GMM parameters for y and y'
        means_y = jnp.stack([1.0 * jnp.ones((d,)), -1.0 * jnp.ones((d,))])  # (2, d)
        stds_y = jnp.stack([0.6 * jnp.ones((d,)), 0.8 * jnp.ones((d,))])  # (2, d)
        weights_y = jnp.array([0.5, 0.5])

        means_yprime = jnp.stack(
            [1.5 * jnp.ones((d,)), -1.5 * jnp.ones((d,))]
        )  # (2, d)
        stds_yprime = jnp.stack([0.5 * jnp.ones((d,)), 0.7 * jnp.ones((d,))])  # (2, d)
        weights_yprime = jnp.array([0.6, 0.4])

        # PRNG
        seed = int(self.run_sett.get("seed", 0))
        key = jax.random.PRNGKey(seed)
        keys = jax.random.split(key, (N + 1) * 2)
        keys_y = keys[: N + 1]
        keys_yprime = keys[N + 1 :]

        y_list = []
        yprime_list = []
        for n in range(N + 1):
            y_n = _sample_gmm(keys_y[n], d, means_y, stds_y, weights_y)
            yprime_n = _sample_gmm(
                keys_yprime[n], d, means_yprime, stds_yprime, weights_yprime
            )
            y_list.append(y_n)
            yprime_list.append(yprime_n)

        y_traj = jnp.stack(y_list, axis=0)  # (N+1, d, 1)
        y_traj_prime = jnp.stack(yprime_list, axis=0)
        return y_traj, y_traj_prime
