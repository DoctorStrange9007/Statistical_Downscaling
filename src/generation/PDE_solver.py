import tensorflow as tf
import numpy as np
import jax.numpy as jnp


def jax_to_tf_wrapper(jax_score_fn):
    """
    Wrapper function to convert TensorFlow tensors to JAX arrays,
    call the JAX function, and convert the result back to TensorFlow tensors.

    Args:
        jax_score_fn: A JAX-based score function that takes (x, t) and returns gradients

    Returns:
        A function compatible with TensorFlow operations
    """

    def numpy_fn(x_np, t_np):
        """Numpy function that can be called from tf.py_function"""
        # Convert numpy arrays to JAX arrays
        x_jax = jnp.array(x_np)
        t_jax = jnp.array(t_np)

        # Call the JAX function
        result_jax = jax_score_fn(x_jax, t_jax)

        # Convert JAX result back to numpy
        result_np = np.array(result_jax)
        # INSERT_YOUR_CODE
        if np.any(np.isinf(result_np)) or (
            np.any(np.isinf(result_np.astype(np.float32)))
        ):
            print("Warning: Inf detected in result_np in jax_to_tf_wrapper.numpy_fn")
            print(f"Inf count: {np.sum(np.isinf(result_np))}")

        return result_np.astype(np.float32)

    def tf_compatible_score(x_tf, t_tf):
        # Use tf.py_function to call the numpy function
        result = tf.py_function(func=numpy_fn, inp=[x_tf, t_tf], Tout=tf.float32)

        # Set the shape explicitly since tf.py_function doesn't preserve shapes
        result.set_shape(x_tf.shape)

        if tf.reduce_any(tf.math.is_inf(result)):
            print("Warning: Inf detected in tensor")
            print(
                f"Inf count: {tf.reduce_sum(tf.cast(tf.math.is_inf(result), tf.int32))}"
            )
            print("ja")

        return result

    return tf_compatible_score


class PDE_solver:
    def __init__(self, model, grad_log, samples, settings):
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
        self.C = np.array(settings["pde_solver"]["C"])
        self.y = samples @ self.C.T
        self.model = model
        # Wrap the JAX-based grad_log function to be compatible with TensorFlow
        self.grad_log = jax_to_tf_wrapper(grad_log)

    def sampler(self):
        """Sample time-space points from the function's domain; points are sampled
            uniformly on the interior of the domain, at the initial/terminal time points
            and along the spatial boundary at different time points.

        Args:
            nSim_interior: number of space points in the interior of the function's domain to sample
            nSim_terminal: number of space points at terminal time to sample (terminal condition)

        # COULD DO:  x_interior = np.random.lognormal(mean=mu, sigma=sigma, size=[nSim_interior, d])
        """

        # Sampler #1: domain interior
        t_interior = np.random.uniform(
            low=self.t_low, high=self.T, size=[self.nSim_interior, 1]
        )
        x_interior = np.random.uniform(
            low=self.x_low,
            high=self.x_high * self.x_multiplier,
            size=[self.nSim_interior, self.d],
        )

        # Sampler #2: spatial boundary
        # no spatial boundary condition for this problem

        # Sampler #3: initial/terminal condition
        t_terminal = self.T * np.ones((self.nSim_terminal, 1))
        x_terminal = np.random.uniform(
            low=self.x_low,
            high=self.x_high * self.x_multiplier,
            size=[self.nSim_terminal, self.d],
        )

        return t_interior, x_interior, t_terminal, x_terminal

    def loss_fn(self, t_interior, x_interior, t_terminal, x_terminal):
        """Compute total loss for training.

        Args:
            model:      DGM model object
            t_interior: sampled time points in the interior of the function's domain
            x_interior: sampled space points in the interior of the function's domain
            t_terminal: sampled time points at terminal time
            x_terminal: sampled space points at terminal time
            C:        C matrix for the linear transformation
            y:        y vector for the linear transformation
            b_bar:    b_bar function
            sigma_bar: sigma_bar function
            T: terminal time
        Returns:
            L1: PDE loss
            L3: terminal condition loss
        """

        # Convert to tensors
        t_interior_tnsr = tf.convert_to_tensor(t_interior, dtype=tf.float32)
        x_interior_tnsr = tf.convert_to_tensor(x_interior, dtype=tf.float32)
        t_terminal_tnsr = tf.convert_to_tensor(t_terminal, dtype=tf.float32)
        x_terminal_tnsr = tf.convert_to_tensor(x_terminal, dtype=tf.float32)

        # Loss term #1: PDE
        # We need to compute derivatives of the option value function V(t,S) with respect to t and S

        with tf.GradientTape(persistent=True) as tape:
            # Watch the input tensors for gradient computation
            tape.watch(t_interior_tnsr)
            tape.watch(x_interior_tnsr)

            # Compute option value V(t,S)
            V = self.model(t_interior_tnsr, x_interior_tnsr)

            # Compute first derivatives
            V_t = tape.gradient(V, t_interior_tnsr)
            V_x = tape.gradient(V, x_interior_tnsr)

            # Compute Hessian using second-order gradients
            # For each component of x, compute gradient of V_x
            V_xx_components = []
            for j in range(x_interior_tnsr.shape[1]):
                V_xx_row = tape.gradient(V_x[:, j], x_interior_tnsr)
                V_xx_components.append(V_xx_row)

            V_xx = tf.stack(V_xx_components, axis=2)  # Shape: (1000, 3, 3)

            # Compute other functions
            b_bar = self.b_bar(t_interior_tnsr, x_interior_tnsr)
            sigma_bar = self.sigma_bar(t_interior_tnsr, self.d, self.T)
            sigma_bar_T = tf.transpose(sigma_bar, perm=[0, 2, 1])

        # PDE residual: V_t + b_bar^T*V_x + 0.5*Tr(sigma*sigma^T*V_xx)
        PDE_residual = (
            V_t
            + tf.reduce_sum(b_bar * V_x, axis=1, keepdims=True)
            + 0.5
            * tf.expand_dims(
                tf.linalg.trace(tf.matmul(tf.matmul(sigma_bar, sigma_bar_T), V_xx)),
                axis=1,
            )
        )

        L1 = tf.reduce_mean(tf.square(PDE_residual))

        if tf.math.is_inf(L1):
            print("ja")

        # Loss term #3: terminal condition
        # At terminal time T, V(T,S) = max(S-K, 0)
        V_terminal = self.model(t_terminal_tnsr, x_terminal_tnsr)
        V_terminal_exact = tf.expand_dims(
            tf.exp(-tf.norm(x_terminal_tnsr @ self.C.T - self.y, axis=1)), axis=1
        )

        L3 = tf.reduce_mean(tf.square(V_terminal - V_terminal_exact))

        return L1, L3

    def train(self):
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        # for each sampling stage
        for i in range(self.sampling_stages):

            # sample uniformly from the required regions
            t_interior, x_interior, t_terminal, x_terminal = self.sampler()

            # for a given sample, take the required number of SGD steps
            for _ in range(self.steps_per_sample):
                with tf.GradientTape() as tape:
                    L1, L3 = self.loss_fn(
                        t_interior, x_interior, t_terminal, x_terminal
                    )
                    loss = L1 + L3
                if tf.math.reduce_any(tf.math.is_nan(L1)) or tf.math.reduce_any(
                    tf.math.is_nan(L3)
                ):
                    print("Warning: L1 or L3 is NaN. Skipping this optimization step.")
                    print(f"L1: {L1.numpy()}, L3: {L3.numpy()}")
                    print(
                        f"t_interior: {t_interior}, x_interior: {x_interior}, t_terminal: {t_terminal}, x_terminal: {x_terminal}"
                    )

                # Compute gradients and apply optimization step
                gradients = tape.gradient(loss, self.model.trainable_variables)
                optimizer.apply_gradients(
                    zip(gradients, self.model.trainable_variables)
                )

            print(
                f"Stage {i}: Loss = {loss.numpy():.6f}, L1 = {L1.numpy():.6f}, L3 = {L3.numpy():.6f}"
            )

    def g_noise(self, t):
        """Noise level function - keeping original numpy implementation"""
        return 0.1 * t + np.sqrt(2) * 0.001

    def alpha(self, t):
        return 0.5

    def b_bar(self, t, x):
        return -self.f(x, 1 - t) + self.grad_log(x, 1 - t) * self.g_noise(1 - t) ** 2

    def sigma_bar(self, t, d, T):
        return self.sigma2(T - t)

    def sigma2(self, t):
        """Convert to TensorFlow operations for compatibility"""
        scaling_factor = 1 - tf.exp(-self.int_beta(t))  # Shape: (batch_size, 1)
        batch_size = tf.shape(scaling_factor)[0]
        identity_matrices = tf.tile(
            tf.expand_dims(tf.eye(self.d), 0), [batch_size, 1, 1]
        )  # Shape: (batch_size, 3, 3)
        result = scaling_factor[:, :, tf.newaxis] * identity_matrices

        return result

    def beta(self, t):
        """Convert to TensorFlow operations for compatibility"""
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def int_beta(self, t):
        """Integral of beta from 0 to t. Convert to TensorFlow operations for compatibility"""
        return t * self.beta_min + 0.5 * t**2 * (self.beta_max - self.beta_min)

    def f(self, x, t):
        """Convert to TensorFlow operations for compatibility"""
        return -0.5 * self.beta(t) * x

    def g_diffusion(self, t):
        """Diffusion coefficient - Convert to TensorFlow operations for compatibility"""
        return tf.sqrt(self.beta(t))

    def s(self, t):
        """Convert to TensorFlow operations for compatibility"""
        return tf.exp(-0.5 * self.int_beta(t))

    def grad_log_h(self, x, t):
        with tf.GradientTape() as tape:
            tape.watch(x)
            h_values = self.model(t, x)
            log_h = tf.math.log(h_values)

        grad_log_h = tape.gradient(log_h, x)

        return grad_log_h
