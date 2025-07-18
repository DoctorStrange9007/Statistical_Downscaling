import numpy as np
from scipy.stats import multivariate_normal
import pandas as pd
import os


class LinearMDP:
    """
    Implements a linear Markov Decision Process (MDP) for statistical downscaling.

    This class provides methods for optimizing mixture model parameters using gradient descent
    to minimize the KL divergence between a parameterized model and a target transition kernel.
    The optimization includes both cost terms and KL divergence regularization.

    The model operates on pairs of states (y, y') and can use different forms parameterized model and target kernel
    (e.g. mixture of Gaussians).
    """

    def __init__(self, run_sett: dict, param_model, trans_kernel):
        """
        Initialize the LinearMDP object.

        Parameters
        ----------
        run_sett : dict
            Dictionary containing model and run settings.
        param_model : object
            Parameter model (e.g., GaussianModel) for the MDP.
        trans_kernel : object
            Transition kernel (e.g., GaussianKernel) for the MDP.
        """
        self.run_sett = run_sett  # Store the full run_sett
        self.model_sett = run_sett["models"]["LinearMDP"]
        self.beta = self.model_sett["beta"]
        self.n_sim = self.model_sett["n_sim"]
        self.nr_gd_steps = self.model_sett["nr_gd_steps"]
        self.d = self.model_sett["d"]
        self.param_model = param_model
        self.trans_kernel = trans_kernel
        self.N = self.model_sett["N"]
        self.converged = [False] * (self.N + 1)
        self.theta_history = []

    def gd(self, lr, save_results=False):
        """
        Run gradient descent optimization for all time steps in the MDP.

        This method performs gradient descent updates on the mixture model parameters
        to minimize the objective function that includes both cost terms and KL divergence
        regularization. The optimization continues until convergence or maximum iterations.

        Parameters
        ----------
        lr : float
            Learning rate for gradient descent updates.
        save_results : bool, optional
            Whether to save results to Excel file (default: False).
        """
        N = self.N
        nr_gd_steps = self.nr_gd_steps

        # Add initial theta values as first row
        if save_results:
            init_data = {"gd_step": -1}  # Use -1 to indicate initial values
            for i, theta_row in enumerate(self.param_model.theta):
                for j, theta_val in enumerate(theta_row):
                    col_name = f"theta_{i}_{j}"
                    init_data[col_name] = theta_val
            self.theta_history.append(init_data)

        print(self.param_model.theta)
        for t in range(nr_gd_steps):
            for n in range(N + 1):
                if n == 0:
                    if not self.converged[0]:
                        self.gd_0(lr)
                else:
                    if not self.converged[n]:
                        self.gd_n(lr, n)
            # Record theta values at this step
            if save_results:
                step_data = {"gd_step": t}
                for i, theta_row in enumerate(self.param_model.theta):
                    for j, theta_val in enumerate(theta_row):
                        col_name = f"theta_{i}_{j}"
                        step_data[col_name] = theta_val
                self.theta_history.append(step_data)

            print(t)
            print("theta")
            print("--------------------------------")
            print(self.param_model.theta)
            print("--------------------------------")
            if np.all(self.converged):
                print("Converged")
                if save_results:
                    self.save_results()
                break
        if save_results:
            self.save_results()

    def save_results(self):
        """
        Save the results of the gradient descent optimization.
        """
        output_dir = self.run_sett["output_dir"]
        df = pd.DataFrame(self.theta_history)
        excel_path = os.path.join(output_dir, "theta_gradient_descent_history.xlsx")
        df.to_excel(excel_path, index=False)
        print(f"Gradient descent history saved to: {excel_path}")

    def gd_0(self, lr):
        """
        Perform one GD update for the initial time step (n=0).

        Parameters
        ----------
        lr : float
            Learning rate for GD update.
        """
        grad = self.compute_gradient_0()
        self.param_model.w[0] -= lr * grad
        new_theta = self.param_model.batched_softmax(self.param_model.w[0])
        if np.allclose(new_theta, self.param_model.theta[0], rtol=0, atol=1e-5):
            self.converged[0] = True
        self.param_model.theta[0] = new_theta

    def gd_n(self, lr, n):
        """
        Perform one GD update for time step n > 0.

        Parameters
        ----------
        lr : float
            Learning rate for  GD update.
        n : int
            Time step index.
        """
        grad = self.compute_gradient_n(n)
        self.param_model.w[n] -= lr * grad
        new_theta = self.param_model.batched_softmax(self.param_model.w[n])
        if np.allclose(new_theta, self.param_model.theta[n], rtol=0, atol=1e-5):
            self.converged[n] = True
        self.param_model.theta[n] = new_theta

    def compute_gradient_0(self):
        """
        Compute the gradient of the objective function with respect to parameters at n=0.

        This method computes the gradient using Monte Carlo sampling and includes
        both the cost term and KL divergence regularization. The gradient is computed
        using the score function estimator.

        Returns
        -------
        np.ndarray
            Gradient vector for the initial 0 time step parameters (theta_0).
        """
        grads = []
        beta = self.beta
        n_sim = self.n_sim
        for b in range(n_sim):
            y0, y0_prime = self.param_model.get_ypair(n=0)
            grad_b = self.U_n(0, yn=y0, yn_prime=y0_prime) * (
                self.param_model.phi_0(y0, y0_prime)
                / self.param_model.q_0(y0, y0_prime)
            )
            grads.append(grad_b)

        grad = np.mean(grads, axis=0) + beta * (
            self.gradient_KL_0() + self.gradient_KL_0_prime()
        )

        return self.jacobian_softmax_n(0) @ grad

    def compute_gradient_n(self, n):
        """
        Compute the gradient of the loss with respect to the parameters at time step n > 0, for n = 1, ..., N.

        Parameters
        ----------
        n : int
            Time step index.

        Returns
        -------
        np.ndarray
            Gradient vector for time step n (theta_n).
        """
        grads = []
        n_sim = self.n_sim
        beta = self.beta

        for b in range(n_sim):
            trajectories = self.param_model.get_trajectories(k=n)
            grad_b = self.U_n(
                n=n, yn=trajectories[n, 0, :], yn_prime=trajectories[n, 1, :]
            ) * (
                self.param_model.phi_n(
                    yn=trajectories[n, 0, :],
                    yn_prime=trajectories[n, 1, :],
                    ynm1=trajectories[n - 1, 0, :],
                    ynm1_prime=trajectories[n - 1, 1, :],
                )
                / self.param_model.q_n(
                    n=n,
                    yn=trajectories[n, 0, :],
                    yn_prime=trajectories[n, 1, :],
                    ynm1=trajectories[n - 1, 0, :],
                    ynm1_prime=trajectories[n - 1, 1, :],
                )
            ) + beta * (
                self.gradient_KL_n_prime(
                    n=n,
                    ynm1=trajectories[n - 1, 0, :],
                    ynm1_prime=trajectories[n - 1, 1, :],
                )
                + self.gradient_KL_n(
                    n=n,
                    ynm1=trajectories[n - 1, 0, :],
                    ynm1_prime=trajectories[n - 1, 1, :],
                )
            )

            grads.append(grad_b)

        return self.jacobian_softmax_n(n=n) @ np.mean(grads, axis=0)

    def gradient_KL_n_prime(self, n, ynm1, ynm1_prime):
        """
        Compute the gradient of the KL divergence with respect to the parameters for the marginal distribution of y'_n.

        Parameters
        ----------
        n : int
            Time step index.
        ynm1 : np.ndarray
            Previous state y_{n-1}.
        ynm1_prime : np.ndarray
            Previous state y'_{n-1}.

        Returns
        -------
        float
            Estimated gradient of the KL divergence for y'_n.
        """
        n_sim = self.n_sim

        def int_func(z):
            return np.log(
                self.param_model.q_n_marg2(n, z, ynm1, ynm1_prime)
                / self.trans_kernel.m_n_prime(n, z, ynm1_prime)
            )

        d = self.model_sett["d"]
        grads = np.zeros(d)
        for i in range(d):
            z_samples = np.zeros((n_sim, d))
            for k in range(n_sim):
                z_samples[k, :] = self.param_model.phi_n_objects(ynm1, ynm1_prime)[1][
                    i
                ].rvs()
            grads[i] = np.mean([int_func(z) for z in z_samples])

        return grads

    def gradient_KL_n(self, n, ynm1, ynm1_prime):
        """
        Compute the gradient of the KL divergence with respect to the parameters for the marginal distribution of y_n.

        Parameters
        ----------
        n : int
            Time step index.
        ynm1 : np.ndarray
            Previous state y_{n-1}.
        ynm1_prime : np.ndarray
            Previous state y'_{n-1}.

        Returns
        -------
        float
            Estimated gradient of the KL divergence for y_n.
        """
        n_sim = self.n_sim

        def int_func(z):
            return np.log(
                self.param_model.q_n_marg1(n, z, ynm1, ynm1_prime)
                / self.trans_kernel.m_n(n, z, ynm1)
            )

        d = self.d
        grads = np.zeros(d)
        for i in range(d):
            z_samples = np.zeros((n_sim, d))
            for k in range(n_sim):
                z_samples[k, :] = self.param_model.phi_n_objects(ynm1, ynm1_prime)[0][
                    i
                ].rvs()
            grads[i] = np.mean([int_func(z) for z in z_samples])

        return grads

    def gradient_KL_0_prime(self):
        """
        Compute the gradient of the KL divergence for the initial marginal distribution of y'_0.

        Returns
        -------
        float
            Estimated gradient of the KL divergence for y'_0.
        """
        n_sim = self.n_sim
        d = self.d

        def int_func(z):
            return np.log(
                self.param_model.q_0_marg2(z) / self.trans_kernel.m_0_prime(z)
            )

        grads = np.zeros(d)
        _, phi_0_objects_y0_prime_temp, _ = self.param_model.phi_0_objects()
        for i in range(d):
            z_samples = np.zeros((n_sim, d))
            for k in range(n_sim):
                z_samples[k, :] = phi_0_objects_y0_prime_temp[i].rvs()
            grads[i] = np.mean([int_func(z) for z in z_samples])

        return grads

    def gradient_KL_0(self):
        """
        Compute the gradient of the KL divergence for the initial marginal distribution of y_0.

        Returns
        -------
        float
            Estimated gradient of the KL divergence for y_0.
        """
        n_sim = self.n_sim
        d = self.model_sett["d"]

        def int_func(z):
            return np.log(self.param_model.q_0_marg1(z) / self.trans_kernel.m_0(z))

        grads = np.zeros(d)
        phi_0_objects_y0_temp, _, _ = self.param_model.phi_0_objects()
        for i in range(d):
            z_samples = np.zeros((n_sim, d))
            for k in range(n_sim):
                z_samples[k, :] = phi_0_objects_y0_temp[i].rvs()
            grads[i] = np.mean([int_func(z) for z in z_samples])

        return grads

    def c_n(self, n, yn, yn_prime):
        """
        Compute the squared Euclidean cost between states yn and yn_prime.

        This function measures the discrepancy between the two state variables
        at time step n. The cost is used in the optimization objective to
        encourage the states to be close to each other.

        Parameters
        ----------
        n : int
            Time step index.
        yn : np.ndarray
            State y_n.
        yn_prime : np.ndarray
            State y'_n.

        Returns
        -------
        float
            Half the squared Euclidean distance between yn and yn_prime.
        """
        return (1 / 2) * np.sum((yn - yn_prime) ** 2)
        # return 0

    def KL_0(self):
        """
        Compute the KL divergence for the initial marginal distribution of y_0.

        Returns
        -------
        float
            Estimated KL divergence for y_0.
        """
        n_sim = self.n_sim
        d = self.model_sett["d"]

        def int_func(z):
            return np.log(self.param_model.q_0_marg1(z) / self.trans_kernel.m_0(z))

        z_samples = np.zeros((n_sim, d))
        for k in range(n_sim):
            i = np.random.choice(d, p=self.param_model.theta[0])
            z_samples[k, :] = self.param_model.phi_0_objects()[0][i].rvs()

        result = np.mean([int_func(z) for z in z_samples])

        return result

    def KL_0_prime(self):
        """
        Compute the KL divergence for the initial marginal distribution of y'_0.

        Returns
        -------
        float
            Estimated KL divergence for y'_0.
        """
        n_sim = self.n_sim
        d = self.d

        def int_func(z):
            return np.log(
                self.param_model.q_0_marg2(z) / self.trans_kernel.m_0_prime(z)
            )

        z_samples = np.zeros((n_sim, d))
        for k in range(n_sim):
            i = np.random.choice(d, p=self.param_model.theta[0])
            z_samples[k, :] = self.param_model.phi_0_objects()[1][i].rvs()

        result = np.mean([int_func(z) for z in z_samples])

        return result

    def KL_n(self, n, ynm1, ynm1_prime):
        """
        Compute the KL divergence for the marginal distribution of y_n.

        Parameters
        ----------
        n : int
            Time step index.
        ynm1 : np.ndarray
            Previous state y_{n-1}.
        ynm1_prime : np.ndarray
            Previous state y'_{n-1}.

        Returns
        -------
        float
            Estimated KL divergence for y_n.
        """
        n_sim = self.n_sim
        d = self.d

        def int_func(z):
            return np.log(
                self.param_model.q_n_marg1(n, z, ynm1, ynm1_prime)
                / self.trans_kernel.m_n(n, z, ynm1)
            )

        z_samples = np.zeros((n_sim, d))
        phi_n_objects_yn_temp, _, _ = self.param_model.phi_n_objects(ynm1, ynm1_prime)
        for k in range(n_sim):
            i = np.random.choice(d, p=self.param_model.theta[n])
            z_samples[k, :] = phi_n_objects_yn_temp[i].rvs()

        result = np.mean([int_func(z) for z in z_samples])

        return result

    def KL_n_prime(self, n, ynm1, ynm1_prime):
        """
        Compute the KL divergence for the marginal distribution of y'_n.

        Parameters
        ----------
        n : int
            Time step index.
        ynm1 : np.ndarray
            Previous state y_{n-1}.
        ynm1_prime : np.ndarray
            Previous state y'_{n-1}.

        Returns
        -------
        float
            Estimated KL divergence for y'_n.
        """
        n_sim = self.n_sim
        d = self.d

        def int_func(z):
            return np.log(
                self.param_model.q_n_marg2(n, z, ynm1, ynm1_prime)
                / self.trans_kernel.m_n_prime(n, z, ynm1_prime)
            )

        z_samples = np.zeros((n_sim, d))
        _, phi_n_objects_yn_prime_temp, _ = self.param_model.phi_n_objects(
            ynm1, ynm1_prime
        )
        for k in range(n_sim):
            i = np.random.choice(d, p=self.param_model.theta[n])
            z_samples[k, :] = phi_n_objects_yn_prime_temp[i].rvs()

        result = np.mean([int_func(z) for z in z_samples])

        return result

    def U_N(self, N, yN, yN_prime):
        """
        Compute the terminal cost at the final time step N.

        Parameters
        ----------
        N : int
            Final time step index.
        yN : np.ndarray
            State y_N.
        yN_prime : np.ndarray
            State y'_N.

        Returns
        -------
        float
            Terminal cost between yN and yN_prime.
        """
        return self.c_n(N, yN, yN_prime)

    def U_n(self, n, yn, yn_prime):
        """
        Compute the cost at time step n, including KL terms and expected future costs.

        Parameters
        ----------
        n : int
            Time step index.
        yn : np.ndarray
            State y_n.
        yn_prime : np.ndarray
            State y'_n.

        Returns
        -------
        float
            Cost at time step n.
        """
        beta = self.beta
        n_sim = self.n_sim
        N = self.N

        if n == N:
            return self.U_N(n, yn, yn_prime)

        exp_ls = []
        for b in range(n_sim):
            exp_inst = 0
            trajectories = self.param_model.get_trajectories(N)

            for k in range(n + 1, N):
                exp_inst += self.c_n(
                    k, yn=trajectories[k, 0, :], yn_prime=trajectories[k, 1, :]
                ) + beta * (
                    self.KL_n(
                        k + 1,
                        ynm1=trajectories[k, 0, :],
                        ynm1_prime=trajectories[k, 1, :],
                    )
                    + self.KL_n_prime(
                        k + 1,
                        ynm1=trajectories[k, 0, :],
                        ynm1_prime=trajectories[k, 1, :],
                    )
                )
            exp_inst += self.c_n(
                N, yn=trajectories[N, 0, :], yn_prime=trajectories[N, 1, :]
            )
            exp_ls.append(exp_inst)
        exp = np.mean(exp_ls)

        return (
            self.c_n(n, yn, yn_prime)
            + beta
            * (self.KL_n(n + 1, yn, yn_prime) + self.KL_n_prime(n + 1, yn, yn_prime))
            + exp
        )

    def jacobian_softmax_n(self, n):
        """
        Compute the Jacobian matrix of the softmax function for time step n.

        This method computes the derivative of the softmax transformation
        with respect to the logits. The Jacobian is used in the chain rule
        when computing gradients with respect to the mixture weights.

        Parameters
        ----------
        n : int
            Time step index.

        Returns
        -------
        np.ndarray
            Jacobian matrix of shape (d, d) for the softmax transformation.
        """
        return np.diag(self.param_model.theta[n]) - np.dot(
            self.param_model.theta[n], self.param_model.theta[n]
        )


class GaussianModel:
    """
    Parameterized mixture of Gaussians model for statistical downscaling.

    This class implements a mixture of multivariate Gaussian distributions
    with learnable mixture weights (theta). The model can represent both
    initial distributions and transition dynamics for pairs of states (y, y').

    The mixture weights are parameterized using softmax transformations
    to ensure they form valid probability distributions.
    """

    def __init__(self, run_sett: dict):
        """
        Initialize the GaussianModel object.

        Parameters
        ----------
        run_sett : dict
            Dictionary containing model and run settings.
        """
        self.model_sett = run_sett["models"]["LinearMDP"]
        self.d = self.model_sett["d"]
        if isinstance(self.model_sett["init_theta"][0][0], str):
            self.theta = np.array(
                [
                    [eval(x) for x in row]
                    for row in np.array(self.model_sett["init_theta"])
                ],
                dtype=float,
            )
        else:
            self.theta = np.array(self.model_sett["init_theta"])
        self.w = self.inverse_softmax(self.theta)

    def batched_softmax(self, w):
        """
        Compute softmax transformation for a vector of logits.

        This function applies the softmax function to convert logits to probability
        distributions. The softmax ensures the output sums to 1 and all values are positive.
        Uses numerically stable implementation to prevent overflow.

        Parameters
        ----------
        w : np.ndarray
            Input logits array of shape (d,) where d is the number of components.

        Returns
        -------
        np.ndarray
            Softmax output of shape (d,) representing probability distribution.
        """
        w = w - np.max(w)  # to avoid overflow

        e = np.exp(w)

        return e / e.sum()

    def inverse_softmax(self, theta):
        """
        Compute logits from probability distribution using inverse softmax.

        This function converts probability distributions back to logits using the
        log transformation. Since the inverse is not unique, we make the logits
        mean-zero to ensure identifiability.

        Parameters
        ----------
        theta : np.ndarray
            Input probability distribution of shape (N, d) or (d,).

        Returns
        -------
        np.ndarray
            Logits of shape (N, d) or (d,) with mean-zero constraint.
        """
        w = np.log(theta)

        # Make it mean-zero to fix identifiability
        w = w - np.mean(w, axis=-1, keepdims=True)

        return w

    def get_ypair(self, n=None, ynm1=None, ynm1_prime=None):
        """
        Sample a pair (y, y') from the model at time n, optionally conditioned on previous states.

        Parameters
        ----------
        n : int, optional
            Time step index.
        ynm1 : np.ndarray, optional
            Previous state y_{n-1}.
        ynm1_prime : np.ndarray, optional
            Previous state y'_{n-1}.

        Returns
        -------
        tuple of np.ndarray
            Sampled states (y, y').
        """
        d = self.d

        if ynm1 is None and ynm1_prime is None:
            i = np.random.choice(d, p=self.theta[0])
            phi_0_objects_y0_temp, phi_0_objects_y0_prime_temp, _ = self.phi_0_objects()
            y0 = phi_0_objects_y0_temp[i].rvs()
            y0_prime = phi_0_objects_y0_prime_temp[i].rvs()
            return y0, y0_prime
        else:
            i = np.random.choice(d, p=self.theta[n])
            phi_n_objects_yn_temp, phi_n_objects_yn_prime_temp, _ = self.phi_n_objects(
                ynm1, ynm1_prime
            )
            yn = phi_n_objects_yn_temp[i].rvs()
            yn_prime = phi_n_objects_yn_prime_temp[i].rvs()
            return yn, yn_prime

    def get_trajectories(self, k):
        """
        Generate trajectories of length k+1 for both y and y' processes.

        This method samples complete trajectories from the mixture model, starting
        from the initial distribution and following the transition dynamics for
        k+1 time steps.

        Parameters
        ----------
        k : int
            Number of transitions (trajectory will have k+1 points).

        Returns
        -------
        np.ndarray
            Trajectory array of shape (k+1, 2, d) where:
            - First dimension: time steps (0 to k)
            - Second dimension: [y, y'] states
            - Third dimension: d-dimensional state space
        """
        d = self.d

        trajectories = np.zeros((k + 1, 2, d))
        for n in range(k + 1):
            if n == 0:
                y0, y0_prime = self.get_ypair(n=0)
                trajectories[n, 0, :] = y0
                trajectories[n, 1, :] = y0_prime
            else:
                y_prev = trajectories[n - 1, 0, :]
                y_prev_prime = trajectories[n - 1, 1, :]
                y, y_prime = self.get_ypair(n=n, ynm1=y_prev, ynm1_prime=y_prev_prime)
                trajectories[n, 0, :] = y
                trajectories[n, 1, :] = y_prime

        return trajectories

    def q_n_marg2(self, n, yn_prime, ynm1, ynm1_prime):
        """
        Compute the marginal probability q_n for y'_n given previous states.

        Parameters
        ----------
        n : int
            Time step index.
        yn_prime : np.ndarray
            State y'_n.
        ynm1 : np.ndarray
            Previous state y_{n-1}.
        ynm1_prime : np.ndarray
            Previous state y'_{n-1}.

        Returns
        -------
        float
            Marginal probability value.
        """
        return np.dot(self.phi_n_marg2(yn_prime, ynm1, ynm1_prime), self.theta[n])

    def q_n_marg1(self, n, yn, ynm1, ynm1_prime):
        """
        Compute the marginal probability q_n for y_n given previous states.

        Parameters
        ----------
        n : int
            Time step index.
        yn : np.ndarray
            State y_n.
        ynm1 : np.ndarray
            Previous state y_{n-1}.
        ynm1_prime : np.ndarray
            Previous state y'_{n-1}.

        Returns
        -------
        float
            Marginal probability value.
        """
        return np.dot(self.phi_n_marg1(yn, ynm1, ynm1_prime), self.theta[n])

    def q_n(self, n, yn, yn_prime, ynm1, ynm1_prime):
        """
        Compute the joint probability q_n for (y_n, y'_n) given previous states.

        Parameters
        ----------
        n : int
            Time step index.
        yn : np.ndarray
            State y_n.
        yn_prime : np.ndarray
            State y'_n.
        ynm1 : np.ndarray
            Previous state y_{n-1}.
        ynm1_prime : np.ndarray
            Previous state y'_{n-1}.

        Returns
        -------
        float
            Joint probability value.
        """
        return np.dot(self.phi_n(yn, yn_prime, ynm1, ynm1_prime), self.theta[n])

    def q_0_marg1(self, y0):
        """
        Compute the marginal probability q_0 for y_0.

        Parameters
        ----------
        y0 : np.ndarray
            State y_0.

        Returns
        -------
        float
            Marginal probability value.
        """
        return np.dot(self.phi_0_marg1(y0), self.theta[0])

    def q_0_marg2(self, y0_prime):
        """
        Compute the marginal probability q_0 for y'_0.

        Parameters
        ----------
        y0_prime : np.ndarray
            State y'_0.

        Returns
        -------
        float
            Marginal probability value.
        """
        return np.dot(self.phi_0_marg2(y0_prime), self.theta[0])

    def q_0(self, y0, y0_prime):
        """
        Compute the joint probability q_0 for (y_0, y'_0).

        Parameters
        ----------
        y0 : np.ndarray
            State y_0.
        y0_prime : np.ndarray
            State y'_0.

        Returns
        -------
        float
            Joint probability value.
        """
        return np.dot(self.phi_0(y0, y0_prime), self.theta[0])

    def phi_n(self, yn, yn_prime, ynm1, ynm1_prime):
        """
        Compute the vector of joint densities for (y_n, y'_n) for all mixture components.

        Parameters
        ----------
        yn : np.ndarray
            State y_n.
        yn_prime : np.ndarray
            State y'_n.
        ynm1 : np.ndarray
            Previous state y_{n-1}.
        ynm1_prime : np.ndarray
            Previous state y'_{n-1}.

        Returns
        -------
        np.ndarray
            Vector of joint densities for each component.
        """
        yyprime = np.concatenate([yn, yn_prime])
        return np.array(
            [
                self.phi_n_objects(ynm1, ynm1_prime)[2][i].pdf(yyprime)
                for i in range(self.d)
            ]
        )

    def phi_n_marg1(self, yn, ynm1, ynm1_prime):
        """
        Compute the vector of marginal densities for y_n for all mixture components.

        Parameters
        ----------
        yn : np.ndarray
            State y_n.
        ynm1 : np.ndarray
            Previous state y_{n-1}.
        ynm1_prime : np.ndarray
            Previous state y'_{n-1}.

        Returns
        -------
        np.ndarray
            Vector of marginal densities for each component.
        """
        return np.array(
            [self.phi_n_objects(ynm1, ynm1_prime)[0][i].pdf(yn) for i in range(self.d)]
        )

    def phi_n_marg2(self, yn_prime, ynm1, ynm1_prime):
        """
        Compute the vector of marginal densities for y'_n for all mixture components.

        Parameters
        ----------
        yn_prime : np.ndarray
            State y'_n.
        ynm1 : np.ndarray
            Previous state y_{n-1}.
        ynm1_prime : np.ndarray
            Previous state y'_{n-1}.

        Returns
        -------
        np.ndarray
            Vector of marginal densities for each component.
        """
        return np.array(
            [
                self.phi_n_objects(ynm1, ynm1_prime)[1][i].pdf(yn_prime)
                for i in range(self.d)
            ]
        )

    def phi_0(self, y0, y0_prime):
        """
        Compute the vector of joint densities for (y_0, y'_0) for all mixture components.

        Parameters
        ----------
        y0 : np.ndarray
            State y_0.
        y0_prime : np.ndarray
            State y'_0.

        Returns
        -------
        np.ndarray
            Vector of joint densities for each component.
        """
        return np.array(
            [
                self.phi_0_objects()[2][i].pdf(np.concatenate([y0, y0_prime]))
                for i in range(self.d)
            ]
        )

    def phi_0_marg1(self, y0):
        """
        Compute the vector of marginal densities for y_0 for all mixture components.

        Parameters
        ----------
        y0 : np.ndarray
            State y_0.

        Returns
        -------
        np.ndarray
            Vector of marginal densities for each component.
        """
        return np.array([self.phi_0_objects()[0][i].pdf(y0) for i in range(self.d)])

    def phi_0_marg2(self, y0_prime):
        """
        Compute the vector of marginal densities for y'_0 for all mixture components.

        Parameters
        ----------
        y0_prime : np.ndarray
            State y'_0.

        Returns
        -------
        np.ndarray
            Vector of marginal densities for each component.
        """
        return np.array(
            [self.phi_0_objects()[1][i].pdf(y0_prime) for i in range(self.d)]
        )

    def phi_n_objects(self, ynm1, ynm1_prime):
        """
        Return the joint and marginal multivariate normal distributions for (y_n, y'_n), y_n, and y'_n for a given component.

        Parameters
        ----------
        ynm1 : np.ndarray
            Previous state y_{n-1}.
        ynm1_prime : np.ndarray
            Previous state y'_{n-1}.
        i : int, optional
            Component index (default: 1).

        Returns
        -------
        tuple
            Tuple of (marginal y_n, marginal y'_n, joint y_n, y'_n) multivariate normal distributions.
        """
        d = self.d

        A = [0.1 ** (i + 1) * np.eye(d) for i in range(d)]
        # B = 0.1*(i+1) * np.eye(d)
        B = [np.zeros((d, d)) for i in range(d)]
        # C = 0.1*(i+1) * np.eye(d)
        C = [np.zeros((d, d)) for i in range(d)]
        D = [0.1 ** (i + 1) * np.eye(d) for i in range(d)]

        Sigma = 0.1 * np.eye(2 * d)

        components_yn = [multivariate_normal(mean=A[i] @ ynm1 + B[i] @ ynm1_prime + np.ones(d) * i * 25, cov=Sigma[:d, :d]) for i in range(d)]  # type: ignore
        components_yn_prime = [multivariate_normal(mean=C[i] @ ynm1 + D[i] @ ynm1_prime + np.ones(d) * i * 25, cov=Sigma[d:, d:]) for i in range(d)]  # type: ignore
        components_yn_yn_prime = [multivariate_normal(mean=np.concatenate([A[i] @ ynm1 + B[i] @ ynm1_prime + np.ones(d) * i * 25, C[i] @ ynm1 + D[i] @ ynm1_prime + np.ones(d) * i * 25]), cov=Sigma) for i in range(d)]  # type: ignore

        return components_yn, components_yn_prime, components_yn_yn_prime

    def phi_0_objects(self):
        """
        Return the joint and marginal multivariate normal distributions for (y_0, y'_0), y_0, and y'_0 for a given component.

        Parameters
        ----------
        i : int, optional
            Component index (default: 1).

        Returns
        -------
        tuple
            Tuple of (marginal y_0, marginal y'_0, joint y_0, y'_0) multivariate normal distributions.
        """
        d = self.d

        A = [0.1 ** (i + 1) * np.ones(d) for i in range(d)]
        B = [0.1 ** (i + 1) * np.ones(d) for i in range(d)]

        Sigma = 0.1 * np.eye(2 * d)

        components_y0 = [multivariate_normal(mean=A[i] + np.ones(d) * i * 25, cov=Sigma[:d, :d]) for i in range(d)]  # type: ignore
        components_y0_prime = [multivariate_normal(mean=B[i] + np.ones(d) * i * 25, cov=Sigma[d:, d:]) for i in range(d)]  # type: ignore
        components_y0_y0_prime = [multivariate_normal(mean=np.concatenate([A[i] + np.ones(d) * i * 25, B[i] + np.ones(d) * i * 25]), cov=Sigma) for i in range(d)]  # type: ignore

        return components_y0, components_y0_prime, components_y0_y0_prime


class GaussianKernel:
    """
    Target transition kernel using mixture of Gaussians.

    This class represents the target distribution that the parameterized model
    should approximate. It uses a fixed mixture of multivariate Gaussian
    distributions with predefined mixture weights (true_theta).

    The kernel generates trajectories according to the target dynamics
    and provides methods for computing probability densities.
    """

    def __init__(self, run_sett: dict):
        """
        Initialize the GaussianKernel object.

        Parameters
        ----------
        run_sett : dict
            Dictionary containing model and run settings.
        """
        self.model_sett = run_sett["models"]["LinearMDP"]
        self.d = self.model_sett["d"]
        if isinstance(self.model_sett["true_theta"][0][0], str):
            self.true_theta = np.array(
                [
                    [eval(x) for x in row]
                    for row in np.array(self.model_sett["true_theta"])
                ],
                dtype=float,
            )
        else:
            self.true_theta = np.array(self.model_sett["true_theta"])

    def MVNMixture_rvs(self, components, n):
        """
        Sample from a mixture of multivariate normal components at time n.

        Parameters
        ----------
        components : list
            List of multivariate normal components.
        n : int
            Time step index.

        Returns
        -------
        np.ndarray
            Sampled value from the mixture.
        """
        d = self.d
        i = np.random.choice(d, p=self.true_theta[n])
        sample = components[i].rvs()
        return sample

    def MVNMixture_pdf(self, components, value, n):
        """
        Compute the PDF of a mixture of multivariate normal components at time n.

        Parameters
        ----------
        components : list
            List of multivariate normal components.
        value : np.ndarray
            Value at which to evaluate the PDF.
        n : int
            Time step index.

        Returns
        -------
        float
            Mixture PDF value at the given value.
        """
        d = self.d

        return np.dot(
            self.true_theta[n], np.array([components[i].pdf(value) for i in range(d)])
        )

    def m_0_objects(self):
        """
        Return the multivariate normal components for y_0 and y'_0.

        Returns
        -------
        tuple
            Tuple of lists of multivariate normal components for y_0 and y'_0.
        """
        d = self.d

        As = [0.1 ** (i + 1) * np.ones(d) for i in range(d)]
        Bs = [0.1 ** (i + 1) * np.ones(d) for i in range(d)]

        Sigma = 0.1 * np.eye(2 * d)

        components_y0 = [multivariate_normal(mean=As[i] + np.ones(d) * i * 25, cov=Sigma[:d, :d]) for i in range(d)]  # type: ignore
        components_y0_prime = [multivariate_normal(mean=Bs[i] + np.ones(d) * i * 25, cov=Sigma[d:, d:]) for i in range(d)]  # type: ignore

        return components_y0, components_y0_prime

    def m_0(self, y0):
        """
        Compute the pdf of a mixture of multivariate normal distributions for y_0.

        Parameters
        ----------
        y0 : np.ndarray
            State y_0.

        Returns
        -------
        float
            Mixture PDF value for y_0.
        """
        return self.MVNMixture_pdf(self.m_0_objects()[0], y0, n=0)

    def m_0_prime(self, y0_prime):
        """
        Compute the pdf of a mixture of multivariate normal distributions for y'_0.

        Parameters
        ----------
        y0_prime : np.ndarray
            State y'_0.

        Returns
        -------
        float
            Mixture PDF value for y'_0.
        """
        return self.MVNMixture_pdf(self.m_0_objects()[1], y0_prime, n=0)

    def m_n_objects(self, n, ynm1=None, ynm1_prime=None):
        """
        Return the multivariate normal components for y_n and/or y'_n at time n.

        Parameters
        ----------
        n : int
            Time step index.
        ynm1 : np.ndarray, optional
            Previous state y_{n-1}.
        ynm1_prime : np.ndarray, optional
            Previous state y'_{n-1}.

        Returns
        -------
        list or tuple
            List(s) of multivariate normal components for y_n and/or y'_n.
        """
        d = self.d

        As = [0.1 ** (i + 1) * np.eye(d) for i in range(d)]
        Ds = [0.1 ** (i + 1) * np.eye(d) for i in range(d)]
        Sigma = 0.1 * np.eye(2 * d)

        if ynm1_prime is None and ynm1 is not None:
            components_yn = [multivariate_normal(mean=As[i] @ ynm1 + np.ones(d) * i * 25, cov=Sigma[:d, :d]) for i in range(d)]  # type: ignore
            return components_yn
        elif ynm1 is None and ynm1_prime is not None:
            components_yn_prime = [multivariate_normal(mean=Ds[i] @ ynm1_prime + np.ones(d) * i * 25, cov=Sigma[d:, d:]) for i in range(d)]  # type: ignore
            return components_yn_prime
        elif ynm1 is not None and ynm1_prime is not None:
            components_yn = [multivariate_normal(mean=As[i] @ ynm1 + np.ones(d) * i * 25, cov=Sigma[:d, :d]) for i in range(d)]  # type: ignore
            components_yn_prime = [multivariate_normal(mean=Ds[i] @ ynm1_prime + np.ones(d) * i * 25, cov=Sigma[d:, d:]) for i in range(d)]  # type: ignore
            return components_yn, components_yn_prime

    def m_n(self, n, yn, ynm1):
        """
        Compute the pdf of a mixture of multivariate normal distributions for y_n at time n.

        Parameters
        ----------
        n : int
            Time step index.
        yn : np.ndarray
            State y_n.
        ynm1 : np.ndarray
            Previous state y_{n-1}.

        Returns
        -------
        float
            Mixture PDF value for y_n.
        """
        return self.MVNMixture_pdf(self.m_n_objects(n, ynm1=ynm1), yn, n=n)

    def m_n_prime(self, n, yn_prime, ynm1_prime):
        """
        Compute the pdf of a mixture of multivariate normal distributions for y'_n at time n.

        Parameters
        ----------
        n : int
            Time step index.
        yn_prime : np.ndarray
            State y'_n.
        ynm1_prime : np.ndarray
            Previous state y'_{n-1}.

        Returns
        -------
        float
            Mixture PDF value for y'_n.
        """
        return self.MVNMixture_pdf(
            self.m_n_objects(n, ynm1_prime=ynm1_prime), yn_prime, n=n
        )
