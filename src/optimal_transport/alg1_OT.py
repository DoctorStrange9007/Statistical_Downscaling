import jax
import jax.numpy as jnp
import jax.lax as lax
from jax.tree_util import tree_map, tree_reduce
from functools import partial
import haiku as hk
import distrax


class PolicyGradient:
    """
    Implements a linear Markov Decision Process (MDP) for statistical downscaling.

    This class provides methods for optimizing mixture model parameters using gradient descent
    to minimize the KL divergence between a parameterized model and a target transition kernel.
    The optimization includes both cost terms and KL divergence regularization.

    The model operates on pairs of states (y, y') and can use different forms parameterized model and target kernel
    (e.g. mixture of Gaussians).
    """

    def __init__(self, run_sett: dict, normalizing_flow_model, true_data_model):
        """
        Initialize the PolicyGradient object.
        """
        self.run_sett = run_sett  # Store the full run_sett
        self.beta = self.run_sett["beta"]
        self.K = self.run_sett["K"]
        self.N = self.run_sett["N"]
        self.d_prime = self.run_sett["d_prime"]
        self.true_data_model = true_data_model
        self.normalizing_flow_model = normalizing_flow_model
        self.B = self.run_sett["B"]
        lr_sett = self.run_sett["lrates"]
        self.lrate = float(
            lr_sett[0] if isinstance(lr_sett, (list, tuple)) else lr_sett
        )
        self.d_phi = 2 * self.d_prime + 1

    def _get_baseline_features(
        self, prev_y: jnp.ndarray, prev_yp: jnp.ndarray
    ) -> jnp.ndarray:
        """
        [DEFINITION OF phi_k,n - Eq. 86]
        Constructs the feature vector phi(y_prev, y_prev_prime) = [1, y_prev, y_prev_prime].
        """

        prev_y = jnp.squeeze(prev_y)
        prev_yp = jnp.squeeze(prev_yp)

        # Feature vector: [1 (bias), y_n-1 (d_prime), y'_n-1 (d_prime)]
        return jnp.concatenate([jnp.ones(1), prev_y, prev_yp])

    def fit_baselines(self, V, S, Phi):
        """
        Fits the optimal constant baseline (n=0, Eq. 88) and linear baselines (n=1..N, Eq. 87)
        using the closed-form solutions derived from least-squares minimization.
        """

        V0, S0 = V[0], S[0]

        c_numerator = jnp.sum(V0 * S0)
        c_optimal = c_numerator / jnp.sum(S0)

        V_stack = jnp.stack(V[1:])
        S_stack = jnp.stack(S[1:])
        Phi_stack = jnp.stack(Phi)

        # A = sum_b (S * phi * phi^T) -> shape (N, D_phi, D_phi)
        A = jnp.einsum("nbi,nb,nbj->nij", Phi_stack, S_stack, Phi_stack)

        # b = sum_b (S * phi * V) -> shape (N, D_phi)
        b_vec = jnp.einsum("nbi,nb,nb->ni", Phi_stack, S_stack, V_stack)

        w_optimal = jnp.linalg.solve(A, b_vec[..., None]).squeeze(-1)

        return {"c": c_optimal, "w": w_optimal}

    def per_trajectory_joint_score(self, n, y_traj, y_traj_prime, params_trees):
        """
        Compute gradient of q_n density w.r.t. params at index n, for a single trajectory.
        Uses JAX control flow to support tracer n from vmap.
        """

        def make_branch(k: int):
            if k == 0:

                def branch(_op=None, k=k):
                    action = jnp.concatenate(
                        [y_traj[0].reshape(-1), y_traj_prime[0].reshape(-1)], axis=0
                    )

                    return self.normalizing_flow_model.get_log_prob_grad(
                        params_trees[k], k, action=action, state=None
                    )

                return branch
            else:

                def branch(_op=None, k=k):
                    state = jnp.concatenate(
                        [y_traj[k - 1].reshape(-1), y_traj_prime[k - 1].reshape(-1)],
                        axis=0,
                    )
                    action = jnp.concatenate(
                        [y_traj[k].reshape(-1), y_traj_prime[k].reshape(-1)],
                        axis=0,
                    )

                    return self.normalizing_flow_model.get_log_prob_grad(
                        params_trees[k], k, action=action, state=state
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
        # return (1/2)*jnp.sum(jnp.abs(y_n_to_N - y_n_prime_to_N))

        return 1.0

    @partial(jax.jit, static_argnums=0)
    def collect_fitting_data_per_trajectory(self, key, params_trees):
        y_traj, y_traj_prime = self.normalizing_flow_model.sample_trajectory(
            key, params_trees
        )

        grads_all = [tree_map(jnp.zeros_like, p) for p in params_trees]

        raw_grads = []
        costs_V = []
        phi_features = []

        for k in range(self.N + 1):
            grad_k = self.per_trajectory_joint_score(
                k, y_traj, y_traj_prime, params_trees
            )
            raw_grads.append(grad_k)

            cost_k = self.per_trajectory_joint_cost(k, y_traj, y_traj_prime)
            costs_V.append(cost_k)

            if k > 0:
                prev_y = y_traj[k - 1]
                prev_yp = y_traj_prime[k - 1]
                phi = self._get_baseline_features(prev_y, prev_yp)
                phi_features.append(phi)

        return tuple(raw_grads), tuple(costs_V), tuple(phi_features)

    def G_val(self, key, params_trees):
        keys = jax.random.split(key, self.B)
        raw_grads_batch, cost_V_batch, phi_features_batch = jax.vmap(
            self.collect_fitting_data_per_trajectory, in_axes=(0, None)
        )(keys, params_trees)

        cost_V_batch = [jnp.array(v) for v in cost_V_batch]
        phi_features_batch = [jnp.array(p) for p in phi_features_batch]

        S_norm_sq_batch = []

        for t in range(self.N + 1):
            grad_t = raw_grads_batch[t]
            norm_sq_score = tree_reduce(
                lambda a, x: a + jnp.sum(x**2, axis=tuple(range(1, x.ndim))),
                grad_t,
                0.0,
            )
            S_norm_sq_batch.append(norm_sq_score)

        baselines = self.fit_baselines(
            cost_V_batch, S_norm_sq_batch, phi_features_batch
        )

        final_grads_sum = []
        for k in range(self.N + 1):
            if k == 0:
                baseline_values = baselines["c"]
            else:
                w_k = baselines["w"][k - 1]
                phi_k = phi_features_batch[k - 1]
                baseline_values = jnp.dot(phi_k, w_k)

            advantages = cost_V_batch[k] - baseline_values

            def scale_grad(g, adv):
                target_shape = (g.shape[0],) + (1,) * (g.ndim - 1)
                return g * adv.reshape(target_shape)

            weighted_grads = tree_map(
                lambda g: scale_grad(g, advantages), raw_grads_batch[k]
            )

            grads_sum = tree_map(lambda g: g.sum(axis=0), weighted_grads)

            final_grads_sum.append(grads_sum)

        return final_grads_sum

    @partial(jax.jit, static_argnums=0)
    def G_KL_run_per_trajectory(self, key, params_trees):
        z_traj, z_traj_prime = self.true_data_model.sample_true_trajectory(key)
        grads_all = [tree_map(jnp.zeros_like, p) for p in params_trees]
        for k in range(self.N + 1):
            grad_k = self.per_trajectory_joint_score(
                k, z_traj, z_traj_prime, params_trees
            )
            grads_all[k] = tree_map(lambda a, b: a + b, grads_all[k], grad_k)

        return tuple(grads_all)

    @partial(jax.jit, static_argnums=0)
    def G_KL(self, key, params_trees):

        keys = jax.random.split(key, self.B)
        result = jax.vmap(self.G_KL_run_per_trajectory, in_axes=(0, None))(
            keys, params_trees
        )
        grads_sum = [
            tree_map(lambda x: -2 * x.sum(axis=0), result[i]) for i in range(self.N + 1)
        ]
        return grads_sum

    @partial(jax.jit, static_argnums=0)
    def aggregate_g(self, key, params_trees):
        key_model, key_true = jax.random.split(key)
        g_val = self.G_val(key_model, params_trees)
        g_kl = self.G_KL(key_true, params_trees)
        # Scale by batch size and beta
        return [
            tree_map(
                lambda gv, gk: (1 / self.B) * gv + (self.beta / self.B) * gk, gv, gk
            )
            for gv, gk in zip(g_val, g_kl)
        ]

    @partial(jax.jit, static_argnums=0)
    def compute_updates(self, params_trees, grads_trees, lrate):
        return [
            tree_map(lambda p, g: p - lrate * g, p_tree, g_tree)
            for p_tree, g_tree in zip(params_trees, grads_trees)
        ]

    def update_params(self, key):

        grads = self.aggregate_g(key, self.normalizing_flow_model.params_trees)
        updated_trees = self.compute_updates(
            self.normalizing_flow_model.params_trees, grads, self.lrate
        )
        self.normalizing_flow_model.params_trees = updated_trees


class NormalizingFlowModel:
    """
    Distrax-based RealNVP flows (unconditional for q0, conditional for qn).
    """

    def __init__(self, run_sett: dict):
        self.run_sett = run_sett
        self.N = self.run_sett["N"]
        self.d_prime = self.run_sett["d_prime"]
        self.state_action_dim = 2 * self.d_prime
        self.num_layers = self.run_sett["num_layers"]
        self.batch_size = self.run_sett["B"]
        self.hidden_size = self.run_sett["hidden_size"]

        self.unc_log_prob = hk.transform(lambda x: self._build_unc_dist().log_prob(x))
        self.unc_sample = hk.transform(
            lambda: self._build_unc_dist().sample(seed=hk.next_rng_key())
        )

        self.cond_log_prob = hk.transform(
            lambda x, s: self._build_cond_dist(s).log_prob(x)
        )
        self.cond_sample = hk.transform(
            lambda s: self._build_cond_dist(s).sample(seed=hk.next_rng_key())
        )

        self.params_trees = []
        self.init_all_models(jax.random.PRNGKey(self.run_sett["seed"]))

    def _build_unc_dist(self):
        """Internal: Builds the Unconditional Distribution (inside transform)."""
        layers = []
        for i in range(self.num_layers):
            mask = jnp.arange(0, self.state_action_dim) % 2 == (i % 2)

            def conditioner(x):
                dummy_state = jnp.zeros_like(
                    x
                )  # we need same shape as n>0 for grad calculations with lax.scan
                model_input = jnp.concatenate([x, dummy_state], axis=-1)
                mlp = hk.Sequential(
                    [
                        hk.Linear(self.hidden_size),
                        jax.nn.tanh,
                        hk.Linear(self.hidden_size),
                        jax.nn.tanh,
                        hk.Linear(self.state_action_dim * 2, w_init=jnp.zeros),
                        hk.Reshape((self.state_action_dim, 2)),
                    ]
                )
                params = mlp(model_input)
                shift = params[..., 0]
                s_pre = params[..., 1]

                scale = jnp.exp(jnp.tanh(s_pre))

                return shift, scale

            layers.append(
                distrax.MaskedCoupling(
                    mask=mask,
                    bijector=lambda params: distrax.ScalarAffine(
                        shift=params[0], scale=params[1]
                    ),
                    conditioner=conditioner,
                )
            )
        base = distrax.MultivariateNormalDiag(
            jnp.zeros(self.state_action_dim), jnp.ones(self.state_action_dim)
        )
        return distrax.Transformed(base, distrax.Chain(layers))

    def _build_cond_dist(self, prev_state):
        """Internal: Builds the Conditional Distribution (inside transform)."""
        layers = []
        for i in range(self.num_layers):
            mask = jnp.arange(0, self.state_action_dim) % 2 == (i % 2)

            def conditioner(x):
                model_input = jnp.concatenate([x, prev_state], axis=-1)
                mlp = hk.Sequential(
                    [
                        hk.Linear(self.hidden_size),
                        jax.nn.tanh,
                        hk.Linear(self.hidden_size),
                        jax.nn.tanh,
                        hk.Linear(self.state_action_dim * 2, w_init=jnp.zeros),
                        hk.Reshape((self.state_action_dim, 2)),
                    ]
                )
                params = mlp(model_input)

                shift = params[..., 0]
                s_pre = params[..., 1]

                scale = jnp.exp(jnp.tanh(s_pre))

                return shift, scale

            layers.append(
                distrax.MaskedCoupling(
                    mask=mask,
                    bijector=lambda params: distrax.ScalarAffine(
                        shift=params[0], scale=params[1]
                    ),
                    conditioner=conditioner,
                )
            )
        base = distrax.MultivariateNormalDiag(
            jnp.zeros(self.state_action_dim), jnp.ones(self.state_action_dim)
        )
        return distrax.Transformed(base, distrax.Chain(layers))

    def init_all_models(self, master_rng):
        """
        Initializes N+1 separate sets of parameters.
        """

        dummy_state = jnp.zeros((1, self.state_action_dim))
        dummy_action = jnp.zeros((1, self.state_action_dim))

        for t in range(self.N + 1):
            master_rng, init_key = jax.random.split(master_rng)
            if t == 0:
                params = self.unc_log_prob.init(init_key, dummy_action)
            else:
                params = self.cond_log_prob.init(init_key, dummy_action, dummy_state)
            self.params_trees.append(params)

    def log_prob_0(self, params, action):
        return self.unc_log_prob.apply(params, None, action)

    def log_prob_cond(self, params, action, state):
        return self.cond_log_prob.apply(params, None, action, state)

    def _sample_0(self, params, key):
        return self.unc_sample.apply(params, key)

    def _sample_cond(self, params, key, state):
        return self.cond_sample.apply(params, key, state)

    def get_log_prob_grad(self, params, n, action, state=None):

        def forward_loss(p):
            if n == 0:
                log_prob = self.log_prob_0(p, action)
            else:
                log_prob = self.log_prob_cond(p, action, state)

            return jnp.mean(log_prob)

        return jax.grad(forward_loss)(params)

    @partial(jax.jit, static_argnums=0)
    def sample_trajectory(self, key, params_trees):
        """
        Sample an entire trajectory using the flow models.
        Returns y_traj, y_traj_prime with shape (N+1, d_prime, 1).
        """
        d = self.d_prime
        key, k0 = jax.random.split(key)

        joint0 = self._sample_0(params_trees[0], k0)
        y0 = joint0[:d].reshape((d, 1))
        y0p = joint0[d:].reshape((d, 1))

        params_stack = tree_map(lambda *args: jnp.stack(args), *params_trees[1:])

        def step(carry, param_t):
            prev_y, prev_yp, key_in = carry
            key_in, ks = jax.random.split(key_in)
            ctx = jnp.concatenate([prev_y.reshape(-1), prev_yp.reshape(-1)], axis=0)
            joint_k = self._sample_cond(param_t, ks, ctx)
            yk = joint_k[:d].reshape((d, 1))
            ypk = joint_k[d:].reshape((d, 1))
            return (yk, ypk, key_in), (yk, ypk)

        init_carry = (y0, y0p, key)
        (_, _, _), (ys, yps) = lax.scan(step, init_carry, xs=params_stack)
        y_traj = jnp.concatenate([y0[None, ...], ys], axis=0)
        y_traj_prime = jnp.concatenate([y0p[None, ...], yps], axis=0)
        return y_traj, y_traj_prime


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
        Pure, key-driven sampler for the true trajectory (JAX-friendly), where
        the component means depend on the previous observation.
        Returns (y_traj, y_traj_prime) with shape (N+1, d_prime, 1).
        """
        d = self.d_prime
        N = self.N
        K = d  # number of mixture components
        scales = 0.5 * jnp.ones((d,))
        weights = jnp.ones((K,)) / float(K)

        # Per-step GMM sampler with dynamic means/stds
        def _sample_gmm_key(k, means, stds, weights):
            k1, k2 = jax.random.split(k)
            comp = jax.random.categorical(k1, jnp.log(weights))
            mean = means[comp]  # (d,)
            std = stds[comp]  # (d,)
            z = jax.random.normal(k2, shape=(d,))
            return (mean + std * z).reshape((d, 1))

        # Initial previous observations set to zeros
        key, ky0, ky0p = jax.random.split(key, 3)
        prev = jnp.zeros((d, 1))
        base_means = jnp.stack(
            [jnp.linspace(-2, 2, d) for _ in range(K)]
        )  # Shape (K, d)
        means_y0 = base_means + prev.reshape(1, d) * 0.5
        stds_y0 = jnp.repeat(scales[None, :], K, axis=0)
        y0 = _sample_gmm_key(ky0, means_y0, stds_y0, weights)

        means_y0p = base_means + prev.reshape(1, d) * 0.5
        stds_y0p = jnp.repeat(scales[None, :], K, axis=0)
        y0p = _sample_gmm_key(ky0p, means_y0p, stds_y0p, weights)

        def step(carry, k_in):
            prev_y, prev_yp, key_c = carry
            key_c, ky, kyp = jax.random.split(key_c, 3)
            means_y = base_means + 0.8 * prev_y.reshape(1, d)
            stds_y = jnp.repeat(scales[None, :], K, axis=0)
            y = _sample_gmm_key(ky, means_y, stds_y, weights)
            means_yp = base_means + 0.8 * prev_yp.reshape(1, d)
            stds_yp = jnp.repeat(scales[None, :], K, axis=0)
            yp = _sample_gmm_key(kyp, means_yp, stds_yp, weights)
            return (y, yp, key_c), (y, yp)

        keys = jax.random.split(key, N)
        (_, _, _), (ys, yps) = lax.scan(step, (y0, y0p, key), xs=keys)
        y_traj = jnp.concatenate([y0[None, ...], ys], axis=0)  # (N+1, d, 1)
        y_traj_prime = jnp.concatenate([y0p[None, ...], yps], axis=0)  # (N+1, d, 1)
        return y_traj, y_traj_prime
