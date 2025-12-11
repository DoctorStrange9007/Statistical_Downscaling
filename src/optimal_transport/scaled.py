import jax
import jax.numpy as jnp
import jax.lax as lax
from jax.tree_util import tree_map, tree_reduce
from functools import partial
import haiku as hk
import distrax
import optax


class PolicyGradient:
    """
    PolicyGradient class for the optimal transport problem.

    This class computes gradients for a sequence of normalizing-flow models
    defining joint densities q_0, q_1, ..., q_N over pairs (y_n, y'_n).
    The total objective consists of:
      - A cost term based on a joint quadratic cost function over trajectory
        segments (n..N), and
      - A KL-based regularizer w.r.t. a true data model.

    Gradients are estimated from mini-batches of sampled trajectories using
    control variates. For n >= 1, the control variate is linear in a feature
    vector built from the previous pair (y_{n-1}, y'_{n-1}); for n = 0 a
    constant control variate is used. Closed-form, weighted least-squares fits
    are used to determine optimal control-variates per time n.
    """

    def __init__(self, run_sett: dict, normalizing_flow_model, true_data_model):
        """
        Initialize the PolicyGradient object.

        Args:
            run_sett: Dictionary of run settings. Must contain:
                - "beta": float, KL regularization weight
                - "N": int, number of transitions (trajectory length is N+1)
                - "d_prime": int, dimension per y (and y')
                - "B": int, batch size for gradient estimation
                - "lrates": float or sequence, learning rate(s); if a sequence,
                  the first element is used here
            normalizing_flow_model: Model exposing:
                - params_trees: list of N+1 parameter pytrees
                - sample_trajectory(key, params_trees) -> (y_traj, y_traj_prime)
                - get_log_prob_grad(params, n, action, state=None) -> grad pytree
            true_data_model: Model exposing:
                - sample_true_trajectory(key) -> (z_traj, z_traj_prime)
        """
        self.run_sett = run_sett  # Store the full run_sett
        self.N = self.run_sett["N"]
        self.d_prime = self.run_sett["d_prime"]
        self.true_data_model = true_data_model
        self.normalizing_flow_model = normalizing_flow_model
        self.B = self.run_sett["B"]
        self.lrate_schedule = optax.warmup_cosine_decay_schedule(
            init_value=float(run_sett["optimizer"]["init_value"]),
            peak_value=float(run_sett["optimizer"]["peak_value"]),
            warmup_steps=int(run_sett["optimizer"]["warmup_steps"]),
            decay_steps=int(run_sett["optimizer"]["decay_steps"]),
            end_value=float(run_sett["optimizer"]["end_value"]),
        )
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate=self.lrate_schedule),
        )
        self.opt_state = self.optimizer.init(self.normalizing_flow_model.params_trees)
        self.d_phi = 2 * self.d_prime + 1

        start_ramp_step = int(self.run_sett["num_iterations"] * 0.2)
        end_ramp_step = int(self.run_sett["num_iterations"] * 0.8)
        ramp_time = end_ramp_step - start_ramp_step
        self.beta_schedule = optax.join_schedules(
            schedules=[
                optax.constant_schedule(
                    self.run_sett["beta_schedule"]["init_boundary_value"]
                ),
                optax.linear_schedule(
                    self.run_sett["beta_schedule"]["init_boundary_value"],
                    self.run_sett["beta_schedule"]["end_boundary_value"],
                    ramp_time,
                ),
                optax.constant_schedule(
                    self.run_sett["beta_schedule"]["end_boundary_value"]
                ),
            ],
            boundaries=[start_ramp_step, end_ramp_step],
        )
        self._step = 0

    def _get_control_variate_features(
        self, prev_y: jnp.ndarray, prev_yp: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Build the linear control-variate feature vector for time n >= 1.

        Given previous observations (y_{n-1}, y'_{n-1}), returns
        phi_{n} = concat([1, y_{n-1}, y'_{n-1}]).

        Args:
            prev_y: Array with shape (d_prime, 1) or (d_prime,), previous y.
            prev_yp: Array with shape (d_prime, 1) or (d_prime,), previous y'.

        Returns:
            Array with shape (2 * d_prime + 1,), the feature vector
            [1, prev_y, prev_yp].
        """

        prev_y = jnp.squeeze(prev_y)
        prev_yp = jnp.squeeze(prev_yp)

        linear = jnp.concatenate([prev_y, prev_yp])
        quadratic = jnp.concatenate([prev_y**2, prev_yp**2, prev_y * prev_yp])

        return jnp.concatenate([jnp.ones(1), linear, quadratic])

    def fit_control_variates(self, costs_V, grads_S_norm_sq_per_batch, phi_features):
        """
        Fit optimal control-variates per time n using closed-form WLS.

        For n = 0, fits a scalar constant c that minimizes the squared error of the cost w.r.t c.
        For each n = 1..N, fits a weight vector w_n for
        the linear control variate l_n(phi) = phi^T w_n. The weighting per
        sample uses the squared score-norm at time n.

        Args:
            costs_V: Sequence of length N+1. Each entry is an array of shape (B,)
                with the joint cost V_n for each trajectory in the batch.
            grads_S_norm_sq_per_batch: Sequence of length N+1. Each entry is an
                array of shape (B,) with the per-trajectory squared L2 norm of
                the score (gradient of log-prob) at time n.
            phi_features: Sequence of length N. Each entry is an array with shape
                (B, d_phi) containing features for n = 1..N, where
                d_phi = 2 * d_prime + 1.

        Returns:
            Dict with:
              - "c": scalar constant control variate for n = 0
              - "w": array with shape (N, d_phi), linear weights for n = 1..N
        """

        cost_V_0, grad_S_0 = costs_V[0], grads_S_norm_sq_per_batch[0]

        c_numerator = jnp.sum(cost_V_0 * grad_S_0)
        c_optimal = c_numerator / jnp.sum(grad_S_0)

        costs_V_stack = jnp.stack(costs_V[1:])
        grad_S_norm_sq_per_batch_stack = jnp.stack(grads_S_norm_sq_per_batch[1:])
        phi_features_stack = jnp.stack(phi_features)

        # A = sum_b (S phi phi^T)
        A = jnp.einsum(
            "nbi,nb,nbj->nij",
            phi_features_stack,
            grad_S_norm_sq_per_batch_stack,
            phi_features_stack,
        )
        n_steps, d_phi, _ = A.shape
        jitter_num_stab = 1e-6 * jnp.eye(d_phi)
        A = A + jitter_num_stab[None, ...]

        # b = sum_b  (S V phi)
        b_vec = jnp.einsum(
            "nbi,nb,nb->ni",
            phi_features_stack,
            grad_S_norm_sq_per_batch_stack,
            costs_V_stack,
        )

        w_optimal = jnp.linalg.solve(A, b_vec[..., None]).squeeze(-1)

        return {"c": c_optimal, "w": w_optimal}

    def per_trajectory_joint_score(self, n, y_traj, y_traj_prime, params_trees):
        """
        Per-trajectory gradient of log q_n w.r.t. parameters at time n.

        Selects the correct flow (unconditional for n = 0, conditional for
        n >= 1) and returns the gradient of the log-probability density evaluated
        at the (state, action) extracted from a single trajectory.

        Args:
            n: Integer in [0, N], index of the flow.
            y_traj: Array with shape (N+1, d_prime, 1), trajectory of y.
            y_traj_prime: Array with shape (N+1, d_prime, 1), trajectory of y'.
            params_trees: List of N+1 parameter pytrees; params_trees[n] is used.

        Returns:
            Pytree matching the structure of params_trees[n], containing the
            gradient of the log-probability density for this trajectory and time.

        Notes:
            Uses lax.switch so that n can be a traced value inside vmaps.
        """

        def make_branch(n: int):
            if n == 0:

                def branch(_op=None, n=n):
                    action = jnp.concatenate(
                        [y_traj[0].reshape(-1), y_traj_prime[0].reshape(-1)], axis=0
                    )

                    return self.normalizing_flow_model.get_log_prob_grad(
                        params_trees[0], 0, action=action, state=None
                    )

                return branch
            else:

                def branch(_op=None, n=n):
                    state = jnp.concatenate(
                        [y_traj[n - 1].reshape(-1), y_traj_prime[n - 1].reshape(-1)],
                        axis=0,
                    )
                    action = jnp.concatenate(
                        [y_traj[n].reshape(-1), y_traj_prime[n].reshape(-1)],
                        axis=0,
                    )

                    return self.normalizing_flow_model.get_log_prob_grad(
                        params_trees[n], n, action=action, state=state
                    )

                return branch

        branches = [make_branch(n) for n in range(self.N + 1)]
        return lax.switch(n, branches, None)

    def per_trajectory_joint_cost(self, n, y_traj, y_traj_prime):
        """
        Compute the joint quadratic cost from step n through N for one trajectory.

        The cost at time n is:
            V_n = 0.5 * sum_{k=n}^N || y_k - y'_k ||_2^2

        Args:
            n: Integer in [0, N], start index for the accumulated cost.
            y_traj: Array with shape (N+1, d_prime, 1), trajectory of y.
            y_traj_prime: Array with shape (N+1, d_prime, 1), trajectory of y'.

        Returns:
            Scalar array, the accumulated cost V_n for this trajectory.
        """
        y_n_to_N = y_traj[n:]
        y_n_prime_to_N = y_traj_prime[n:]
        return (1 / 2) * jnp.sum(
            (y_n_to_N - y_n_prime_to_N) ** 2
        )  # sums over axes n to N and d_prime

    @partial(jax.jit, static_argnums=0)
    def collect_fitting_data_per_trajectory(self, key, params_trees):
        """
        Collect per-trajectory quantities needed to fit control variates.

        For a single sampled trajectory, returns:
          - The score gradients at each time n (as pytrees),
          - The joint costs V_n for n = 0..N,
          - The feature vectors phi_n for n = 1..N.

        Args:
            key: PRNGKey for sampling the trajectory.
            params_trees: List of N+1 parameter pytrees used for sampling.

        Returns:
            Tuple of (grads_S, costs_V, phi_features) where:
              - grads_S: tuple length N+1; entry n is a pytree of gradients
                for time n (no batch dimension).
              - costs_V: tuple length N+1 of scalars V_n for this trajectory.
              - phi_features: tuple length N of arrays with shape (d_phi,)
                for times 1..N.
        """
        y_traj, y_traj_prime = self.normalizing_flow_model.sample_trajectory(
            key, params_trees
        )

        grads_S = []
        costs_V = []
        phi_features = []

        for n in range(self.N + 1):
            grad_n = self.per_trajectory_joint_score(
                n, y_traj, y_traj_prime, params_trees
            )
            grads_S.append(grad_n)

            cost_n = self.per_trajectory_joint_cost(n, y_traj, y_traj_prime)
            costs_V.append(cost_n)

            if n > 0:
                prev_y = y_traj[n - 1]
                prev_yp = y_traj_prime[n - 1]
                phi_n = self._get_control_variate_features(prev_y, prev_yp)
                phi_features.append(phi_n)

        return tuple(grads_S), tuple(costs_V), tuple(phi_features)

    def G_val(self, key, params_trees):
        """
        Compute G_val -L (adjusted for control variates (note slightly different than notation in document)).

        This function:
          1) Samples B trajectories,
          2) Computes per-time score gradients and costs,
          3) Fits optimal control variates (constant for n=0, linear for n>=1),
          4) Forms advantages (V_n - l_n) and weights gradients accordingly,
          5) Sums gradients over the batch dimension.

        Args:
            key: PRNGKey for trajectory sampling.
            params_trees: List of N+1 parameter pytrees.

        Returns:
            List length N+1 of pytrees. Each pytree matches params_trees[n] and
            contains the batch-summed gradient contribution for time n (not
            averaged by B yet).
        """
        keys = jax.random.split(key, self.B)
        grads_S, costs_V, phi_features = jax.vmap(
            self.collect_fitting_data_per_trajectory, in_axes=(0, None)
        )(keys, params_trees)

        costs_V = [jnp.array(v) for v in costs_V]
        phi_features = [jnp.array(p) for p in phi_features]

        grads_S_norm_sq_per_batch = []

        for n in range(self.N + 1):
            grad_n = grads_S[n]
            # iterates ocer all leaves of the gradient pytree computes squared L2,
            # accumulates per leaf norms across the whole tree starting from 0.0
            # resulting in length B vector with the total squared gradient norm for each batch at time n
            norm_sq_score = tree_reduce(
                lambda a, x: a + jnp.sum(x**2, axis=tuple(range(1, x.ndim))),
                grad_n,
                0.0,
            )
            grads_S_norm_sq_per_batch.append(norm_sq_score)

        control_variates_results = self.fit_control_variates(
            costs_V, grads_S_norm_sq_per_batch, phi_features
        )

        final_grads_sum = []
        for n in range(self.N + 1):
            if n == 0:
                l_n = control_variates_results["c"]
            else:
                w_n = control_variates_results["w"][n - 1]
                phi_n = phi_features[n - 1]
                l_n = jnp.dot(phi_n, w_n)

            diff_V_l = costs_V[n] - l_n

            def scale_grad(g, adv):
                target_shape = (g.shape[0],) + (1,) * (g.ndim - 1)
                return g * adv.reshape(target_shape)

            weighted_grads = tree_map(lambda g: scale_grad(g, diff_V_l), grads_S[n])

            grads_sum = tree_map(lambda g: g.sum(axis=0), weighted_grads)

            final_grads_sum.append(grads_sum)

        return final_grads_sum

    @partial(jax.jit, static_argnums=0)
    def G_KL_run_per_trajectory(self, key, params_trees):
        """
        Per-trajectory contribution of the KL regularizer gradient.

        Draws one trajectory from the true data model and accumulates the
        gradients of the model log-probabilities evaluated along that true
        trajectory for all times n.

        Args:
            key: PRNGKey for sampling the true trajectory.
            params_trees: List of N+1 parameter pytrees.

        Returns:
            Tuple length N+1 of pytrees; entry n is the gradient w.r.t.
            params_trees[n] for this single trajectory.
        """
        z_traj, z_traj_prime = self.true_data_model.sample_true_trajectory(key)
        grads_all = [tree_map(jnp.zeros_like, p) for p in params_trees]
        for n in range(self.N + 1):
            grad_n = self.per_trajectory_joint_score(
                n, z_traj, z_traj_prime, params_trees
            )
            grads_all[n] = tree_map(lambda a, b: a + b, grads_all[n], grad_n)

        return tuple(grads_all)

    @partial(jax.jit, static_argnums=0)
    def G_KL(self, key, params_trees):
        """
        Compute batch-summed gradient of the KL regularizer.

        Samples B true trajectories and sums their per-time gradients, applying
        a factor of -2.

        Args:
            key: PRNGKey for sampling true trajectories.
            params_trees: List of N+1 parameter pytrees.

        Returns:
            List length N+1 of pytrees, where entry n contains the batch-summed
            KL gradient contribution for time n (already scaled by -2).
        """
        keys = jax.random.split(key, self.B)
        result = jax.vmap(self.G_KL_run_per_trajectory, in_axes=(0, None))(
            keys, params_trees
        )
        grads_sum = [
            tree_map(lambda x: -2 * x.sum(axis=0), result[i]) for i in range(self.N + 1)
        ]
        return grads_sum

    @partial(jax.jit, static_argnums=0)
    def g(self, key, params_trees, beta_value):
        """
        Full gradient estimator combining cost and KL terms.

        Computes:
            (1/B) * (G_val(key_model) - L) + (beta/B) * G_KL(key_true)
        where G_val and G_KL each return batch-summed gradients. The result is
        therefore a batch-averaged gradient per time n.

        Args:
            key: PRNGKey split into model and true-data keys.
            params_trees: List of N+1 parameter pytrees.
            beta_value: Scalar beta used to scale the KL term at this step.

        Returns:
            List length N+1 of pytrees containing the averaged gradient estimate
            for times n = 0..N.
        """
        key_model, key_true = jax.random.split(key)
        G_val = self.G_val(
            key_model, params_trees
        )  # note that this is actually G_val-L
        G_kl = self.G_KL(key_true, params_trees)
        # Scale by batch size and beta
        return [
            tree_map(
                lambda gv, gk: (1 / self.B) * gv + (beta_value / self.B) * gk, gv, gk
            )
            for gv, gk in zip(G_val, G_kl)
        ]

    def update_params(self, key):
        """
        One optimization step: estimate gradients and update model parameters in-place.

        Splits the key, computes the full gradient estimate via `g`, applies
        clipped gradient descent updates using the configured learning rate,
        and writes the updated parameter trees back into the normalizing-flow model.

        Args:
            key: PRNGKey used for both model and true-data sampling.
        """

        beta_value = self.beta_schedule(self._step)
        grads = self.g(key, self.normalizing_flow_model.params_trees, beta_value)

        updates, self.opt_state = self.optimizer.update(
            grads, self.opt_state, self.normalizing_flow_model.params_trees
        )
        self.normalizing_flow_model.params_trees = optax.apply_updates(
            self.normalizing_flow_model.params_trees, updates
        )
        self._step += 1


class NormalizingFlowModel:
    """
    Distrax-based RealNVP flows (unconditional for q0, conditional for qn).
    """

    def __init__(self, run_sett: dict, true_data_model):
        """
        Construct a family of RealNVP flows for joint state-action vectors.

        This creates two Haiku-transformed callables for both the unconditional
        (n = 0) and conditional (n >= 1) models:
          - log_prob transforms, returning log probability densities
          - sample transforms, returning RNG-driven samples

        Settings pulled from run_sett:
          - N: number of transitions (trajectory length is N+1)
          - d_prime: per-variable dimension (y and y'); state_dim = 2 * d_prime
          - num_layers: number of alternating masked-coupling layers
          - hidden_size: width of the conditioner MLPs
          - seed: PRNG seed used for parameter initialization

        Args:
            run_sett: Dictionary with fields described above.
        """
        self.run_sett = run_sett
        self.N = self.run_sett["N"]
        self.d_prime = self.run_sett["d_prime"]
        self.num_bins = self.run_sett["num_bins"]
        self.state_dim = 2 * self.d_prime
        self.num_layers = self.run_sett["num_layers"]
        self.hidden_size = self.run_sett["hidden_size"]

        self.unc_log_prob = hk.transform(
            lambda x: self._build_unc_dist(n=0).log_prob(x)
        )
        self.unc_sample = hk.transform(
            lambda: self._build_unc_dist(n=0).sample(seed=hk.next_rng_key())
        )

        self.cond_log_prob = hk.transform(
            lambda x, s, n: self._build_cond_dist(s, n).log_prob(x)
        )
        self.cond_sample = hk.transform(
            lambda s, n: self._build_cond_dist(s, n).sample(seed=hk.next_rng_key())
        )

        self.params_trees = []
        self.mu, self.sig = self._calibrate(true_data_model)
        self.init_all_models(jax.random.PRNGKey(self.run_sett["seed"]))

    def _calibrate(self, true_data_model):
        """Generates a batch of data to compute mean and std to normalize data before training."""
        key = jax.random.PRNGKey(self.run_sett["seed"])
        keys = jax.random.split(key, 2000)
        z_trajs, z_trajs_prime = jax.vmap(true_data_model.sample_true_trajectory)(keys)
        joint_data = jnp.concatenate([z_trajs, z_trajs_prime], axis=2)
        # Compute per-time (n) and per-dimension statistics; keep axes 1 (n) and 2 (state_dim)
        mu = jnp.mean(joint_data, axis=(0, 3))
        sig = jnp.std(joint_data, axis=(0, 3))
        sig = sig + 1e-6

        return mu, sig

    def _build_unc_dist(self, n: int):
        """
        Build the unconditional flow distribution q0 over R^{state_dim}.

        Architecture:
          - Chain of `num_layers` RealNVP masked-coupling layers
          - Alternating binary masks across layers
          - Conditioner MLP produces per-dimension (shift, scale) parameters
            with elementwise scale = exp(tanh(s_pre)) for stability
          - Base distribution is standard multivariate normal (diag covariance)

        Returns:
            A `distrax.Transformed` distribution with methods `.log_prob(x)`
            and `.sample(seed=...)` operating on vectors of shape (..., state_dim).
        """
        layers = []
        for i in range(self.num_layers):
            mask = jnp.arange(0, self.state_dim) % 2 == (i % 2)
            num_bins = self.num_bins

            def conditioner(x):
                dummy_state = jnp.zeros_like(x)
                init = hk.initializers.VarianceScaling(
                    scale=1.0, mode="fan_avg", distribution="uniform"
                )
                model_input = jnp.concatenate([x, dummy_state], axis=-1)
                mlp = hk.Sequential(
                    [
                        hk.Linear(self.hidden_size, w_init=init),
                        jax.nn.tanh,
                        hk.Linear(self.hidden_size, w_init=init),
                        jax.nn.tanh,
                        hk.Linear(
                            self.state_dim
                            * (
                                3 * num_bins + 1
                            ),  # NSF (with linear throuws away 2 -> 3K-1)
                            w_init=jnp.zeros,
                            b_init=jnp.zeros,
                        ),
                        hk.Reshape(
                            (self.state_dim, 3 * num_bins + 1)
                        ),  # NSF (with linear throuws away 2 -> 3K-1)
                    ]
                )
                params = mlp(model_input)

                return params

            layers.append(
                distrax.MaskedCoupling(
                    mask=mask,
                    bijector=lambda params: distrax.RationalQuadraticSpline(
                        params,
                        range_min=-4.0,
                        range_max=4.0,
                        boundary_slopes="identity",
                        min_bin_size=1e-3,
                        min_knot_slope=1e-3,
                    ),
                    conditioner=conditioner,
                )
            )
        # De-normalize back to data space: sig[n] * x + mu[n]
        mu_n = self.mu[n]
        sig_n = self.sig[n]
        layers.append(
            distrax.Block(
                distrax.ScalarAffine(shift=mu_n, scale=sig_n),
                ndims=1,
            )
        )
        base = distrax.MultivariateNormalDiag(
            jnp.zeros(self.state_dim), jnp.ones(self.state_dim)
        )
        return distrax.Transformed(base, distrax.Chain(layers))

    def _build_cond_dist(self, prev_state, n: int):
        """
        Build the conditional flow distribution qn(· | prev_state).

        Similar to `_build_unc_dist` but the conditioner takes the previous
        state as additional input to produce per-dimension (shift, scale)
        parameters conditioned on `prev_state`.

        Args:
            prev_state: Array with shape (..., state_dim). Acts as context and
                is concatenated with the masked input inside the conditioner.

        Returns:
            A `distrax.Transformed` conditional distribution whose `.log_prob(x)`
            and `.sample(seed=...)` use the captured `prev_state` context. Inputs
            and outputs use vectors of shape (..., state_dim).
        """
        layers = []
        mu_prev = self.mu[n - 1]
        sig_prev = self.sig[n - 1]
        prev_state_norm = (prev_state - mu_prev) / sig_prev
        for i in range(self.num_layers):
            mask = jnp.arange(0, self.state_dim) % 2 == (i % 2)
            num_bins = self.num_bins

            def conditioner(x):
                init = hk.initializers.VarianceScaling(
                    scale=1.0, mode="fan_avg", distribution="uniform"
                )
                model_input = jnp.concatenate([x, prev_state_norm], axis=-1)
                mlp = hk.Sequential(
                    [
                        hk.Linear(self.hidden_size, w_init=init),
                        jax.nn.tanh,
                        hk.Linear(self.hidden_size, w_init=init),
                        jax.nn.tanh,
                        hk.Linear(
                            self.state_dim
                            * (
                                3 * num_bins + 1
                            ),  # NSF (with linear throuws away 2 -> 3K-1)
                            w_init=jnp.zeros,
                            b_init=jnp.zeros,
                        ),
                        hk.Reshape(
                            (self.state_dim, 3 * num_bins + 1)
                        ),  # NSF (with linear throuws away 2 -> 3K-1)
                    ]
                )
                params = mlp(model_input)

                return params

            layers.append(
                distrax.MaskedCoupling(
                    mask=mask,
                    bijector=lambda params: distrax.RationalQuadraticSpline(
                        params,
                        range_min=-4.0,
                        range_max=4.0,
                        boundary_slopes="identity",
                        min_bin_size=1e-3,
                        min_knot_slope=1e-3,
                    ),
                    conditioner=conditioner,
                )
            )
        # De-normalize action back to data space for step n
        mu_n = self.mu[n]
        sig_n = self.sig[n]
        layers.append(
            distrax.Block(
                distrax.ScalarAffine(shift=mu_n, scale=sig_n),
                ndims=1,
            )
        )
        base = distrax.MultivariateNormalDiag(
            jnp.zeros(self.state_dim), jnp.ones(self.state_dim)
        )
        return distrax.Transformed(base, distrax.Chain(layers))

    def init_all_models(self, master_rng):
        """
        Initialize parameters for q0 and q1..qN and store them in `params_trees`.

        Creates N+1 parameter pytrees by calling the `.init` of the transformed
        Haiku functions:
          - Index 0: parameters for the unconditional log_prob (q0)
          - Indices 1..N: parameters for the conditional log_prob (qn)

        Args:
            master_rng: JAX PRNGKey used to split keys for each initialization.

        Side effects:
            Populates `self.params_trees` with a list of length N+1.
        """

        dummy_state = jnp.zeros((1, self.state_dim))
        dummy_action = jnp.zeros((1, self.state_dim))

        for n in range(self.N + 1):
            master_rng, init_key = jax.random.split(master_rng)
            if n == 0:
                params = self.unc_log_prob.init(init_key, dummy_action)
            else:
                params = self.cond_log_prob.init(init_key, dummy_action, dummy_state, n)
            self.params_trees.append(params)

    def log_prob_0(self, params, action):
        """
        Compute log q0(action) for the unconditional model.

        Args:
            params: Haiku parameter pytree for q0.
            action: Array with shape (..., state_dim), the input vector.

        Returns:
            Array of log-probabilities with shape matching any leading batch dims.
        """
        return self.unc_log_prob.apply(params, None, action)

    def log_prob_cond(self, params, action, state, n):
        """
        Compute log qn(action | state) for the conditional model.

        Args:
            params: Haiku parameter pytree for qn at some step n >= 1.
            action: Array with shape (..., state_dim), the input vector.
            state: Array with shape (..., state_dim), the conditioning vector
                (typically concatenation of previous y and y').
            n: Integer step index selecting per-time normalization.

        Returns:
            Array of log-probabilities with shape matching any leading batch dims.
        """
        return self.cond_log_prob.apply(params, None, action, state, n)

    def _sample_0(self, params, key):
        """
        Sample a vector from the unconditional model q0.

        Args:
            params: Haiku parameter pytree for q0.
            key: JAX PRNGKey used by the transformed sampler.

        Returns:
            Array with shape (state_dim,) or with leading batch dims if batched apply.
        """
        return self.unc_sample.apply(params, key)

    def _sample_cond(self, params, key, state, n):
        """
        Sample a vector from the conditional model qn(· | state).

        Args:
            params: Haiku parameter pytree for qn at some step n >= 1.
            key: JAX PRNGKey used by the transformed sampler.
            state: Array with shape (..., state_dim), conditioning vector.
            n: Integer step in [1..N].

        Returns:
            Array with shape (state_dim,) or with leading batch dims if batched apply.
        """
        return self.cond_sample.apply(params, key, state, n)

    def get_log_prob_grad(self, params, n, action, state=None):
        """
        Gradient of log-probability with respect to parameters at step n.

        For n = 0, computes ∇_params log q0(action).
        For n >= 1, computes ∇_params log qn(action | state).

        Args:
            params: Haiku parameter pytree for the relevant step.
            n: Integer in [0, N], selects unconditional (0) or conditional (>=1).
            action: Array with shape (..., state_dim), input vector.
            state: Optional array with shape (..., state_dim), required if n >= 1.

        Returns:
            Pytree with the same structure as `params`, containing gradients.
        """

        def forward_loss(p):
            if n == 0:
                log_prob = self.log_prob_0(p, action)
            else:
                log_prob = self.log_prob_cond(p, action, state, n)

            return log_prob

        return jax.grad(forward_loss)(params)

    @partial(jax.jit, static_argnums=0)
    def sample_trajectory(self, key, params_trees):
        """
        Sample an entire (y, y') trajectory from q0 and q1..qN.

        Process:
          1) Sample joint0 ~ q0 to obtain (y0, y0')
          2) For n = 1..N, sample joint_n ~ q_n(· | prev_state) where
             prev_state = concat(y_{n-1}, y'_{n-1})
          3) Return stacked arrays for y and y' with explicit singleton channel

        Args:
            key: PRNGKey for sequential sampling across time.
            params_trees: List of N+1 Haiku parameter pytrees; index 0 for q0,
                and indices 1..N for q1..qN.

        Returns:
            Tuple (y_traj, y_traj_prime):
              - y_traj: Array with shape (N+1, d_prime, 1)
              - y_traj_prime: Array with shape (N+1, d_prime, 1)
        """
        d = self.d_prime
        key, k0 = jax.random.split(key)

        joint0 = self._sample_0(params_trees[0], k0)
        y0 = joint0[:d].reshape((d, 1))
        y0p = joint0[d:].reshape((d, 1))

        params_stack = tree_map(lambda *args: jnp.stack(args), *params_trees[1:])
        n_indices = jnp.arange(1, self.N + 1)

        def step(carry, xs):
            param_t, n_idx = xs
            prev_y, prev_yp, key_in = carry
            key_in, ks = jax.random.split(key_in)
            ctx = jnp.concatenate([prev_y.reshape(-1), prev_yp.reshape(-1)], axis=0)
            joint_k = self._sample_cond(param_t, ks, ctx, n_idx)
            yk = joint_k[:d].reshape((d, 1))
            ypk = joint_k[d:].reshape((d, 1))
            return (yk, ypk, key_in), (yk, ypk)

        init_carry = (y0, y0p, key)
        (_, _, _), (ys, yps) = lax.scan(step, init_carry, xs=(params_stack, n_indices))
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
