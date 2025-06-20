import numpy as np
from scipy.stats import multivariate_normal


class LinearMDP:
    """
    Implements a linear Markov Decision Process (MDP) with methods for stochastic gradient descent,
    KL divergence computation, and Monte Carlo integration.
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
        self.model_sett = run_sett["models"]["LinearMDP"]
        self.param_model = param_model
        self.trans_kernel = trans_kernel
        self.beta = self.model_sett["beta"]

    def monte_carlo_integral(self, func, bounds, n_samples=100):
        """
        Estimate a multidimensional integral using Monte Carlo sampling.

        Parameters
        ----------
        func : callable
            Function to integrate. Should accept a 1D numpy array.
        bounds : list of tuple
            List of (lower, upper) bounds for each dimension.
        n_samples : int, optional
            Number of Monte Carlo samples (default: 100).

        Returns
        -------
        float
            Estimated value of the integral.
        """
        d = len(bounds)
        samples = np.random.uniform(
            low=[b[0] for b in bounds], high=[b[1] for b in bounds], size=(n_samples, d)
        )
        vals = np.array([func(sample) for sample in samples])
        volume = np.prod([b[1] - b[0] for b in bounds])
        return np.mean(vals) * volume

    def sgd(self, lr):
        """
        Run gradient descent for all time steps in the MDP.

        Parameters
        ----------
        lr : float
            Learning rate for SGD updates.
        """
        N = self.model_sett["N"]
        nr_sg_steps = self.model_sett["nr_sg_steps"]
        print(self.param_model.theta)
        for t in range(nr_sg_steps):
            for n in range(N + 1):
                if n == 0:
                    self.sgd_0(lr)
                else:
                    self.sgd_n(lr, n)
            print(t)
            print(self.param_model.theta)
        print(self.param_model.theta)

    def sgd_0(self, lr):
        """
        Perform one SGD update for the initial time step (n=0).

        Parameters
        ----------
        lr : float
            Learning rate for SGD update.
        """
        grad = self.compute_gradient_0()
        self.param_model.w[0] -= lr * grad
        self.param_model.theta[0] = self.param_model.batched_softmax(
            self.param_model.w[0]
        )

    def sgd_n(self, lr, n):
        """
        Perform one SGD update for time step n > 0.

        Parameters
        ----------
        lr : float
            Learning rate for SGD update.
        n : int
            Time step index.
        """
        grad = self.compute_gradient_n(n)
        self.param_model.w[n] -= lr * grad
        self.param_model.theta[n] = self.param_model.batched_softmax(
            self.param_model.w[n]
        )

    def compute_gradient_0(self):
        """
        Compute the gradient of the loss with respect to the parameters at n=0.

        Returns
        -------
        np.ndarray
            Gradient vector for n=0.
        """
        grads = []
        beta = self.model_sett["beta"]
        B = self.model_sett["n_sim"]
        for b in range(B):
            y0, y0_prime = self.param_model.get_ypair(n=0)
            grad_b = self.U_n(0, yn=y0, yn_prime=y0_prime) * (
                self.param_model.phi_0(y0, y0_prime)
                / self.param_model.q_0(y0, y0_prime)
            )
            grads.append(grad_b)

        grad = (1 / B) * sum(grads) + beta * (
            self.gradient_KL_0() + self.gradient_KL_0_prime()
        )

        return self.jacobian_softmax_n(0) @ grad

    def compute_gradient_n(self, n):
        """
        Compute the gradient of the loss with respect to the parameters at time step n > 0.

        Parameters
        ----------
        n : int
            Time step index.

        Returns
        -------
        np.ndarray
            Gradient vector for time step n.
        """
        grads = []
        B = self.model_sett["n_sim"]
        beta = self.model_sett["beta"]

        for b in range(B):
            for i in range(n + 1):
                if i == 0:
                    y0, y0_prime = self.param_model.get_ypair(n=0)
                    trajectory = [(y0, y0_prime)]
                else:
                    y_prev, y_prev_prime = trajectory[i - 1]
                    y, y_prime = self.param_model.get_ypair(
                        n=i, ynm1=y_prev, ynm1_prime=y_prev_prime
                    )
                    trajectory.append((y, y_prime))
            grad_b = self.U_n(n, yn=y, yn_prime=y_prime) * (
                self.param_model.phi_n(y, y_prime, y_prev, y_prev_prime)
                / self.param_model.q_n(n, y, y_prime, y_prev, y_prev_prime)
            ) + beta * (
                self.gradient_KL_n_prime(n, y, y_prime)
                + self.gradient_KL_n(n, y, y_prime)
            )

            grads.append(grad_b)

        return self.jacobian_softmax_n(n) @ ((1 / B) * np.sum(grads, axis=0))

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

        def int_func(z):
            return np.log(
                self.param_model.q_n_marg2(n, z, ynm1, ynm1_prime)
                / self.trans_kernel.m_n_prime(n, z, ynm1_prime)
            ) * self.param_model.phi_n_marg2(z, ynm1, ynm1_prime)

        bound = tuple(map(int, self.model_sett["bound"].split(",")))
        bounds = [bound for _ in range(self.model_sett["d"])]
        result = self.monte_carlo_integral(int_func, bounds)
        return result

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

        def int_func(z):
            return np.log(
                self.param_model.q_n_marg1(n, z, ynm1, ynm1_prime)
                / self.trans_kernel.m_n(n, z, ynm1)
            ) * self.param_model.phi_n_marg1(z, ynm1, ynm1_prime)

        bound = tuple(map(int, self.model_sett["bound"].split(",")))
        bounds = [bound for _ in range(self.model_sett["d"])]
        result = self.monte_carlo_integral(int_func, bounds)
        return result

    def gradient_KL_0_prime(self):
        """
        Compute the gradient of the KL divergence for the initial marginal distribution of y'_0.

        Returns
        -------
        float
            Estimated gradient of the KL divergence for y'_0.
        """

        def int_func(z):
            return np.log(
                self.param_model.q_0_marg2(z) / self.trans_kernel.m_0_prime(z)
            ) * self.param_model.phi_0_marg2(z)

        bound = tuple(map(int, self.model_sett["bound"].split(",")))
        bounds = [bound for _ in range(self.model_sett["d"])]
        result = self.monte_carlo_integral(int_func, bounds)
        return result

    def gradient_KL_0(self):
        """
        Compute the gradient of the KL divergence for the initial marginal distribution of y_0.

        Returns
        -------
        float
            Estimated gradient of the KL divergence for y_0.
        """

        def int_func(z):
            return np.log(
                self.param_model.q_0_marg1(z) / self.trans_kernel.m_0(z)
            ) * self.param_model.phi_0_marg1(z)

        bound = tuple(map(int, self.model_sett["bound"].split(",")))
        bounds = [bound for _ in range(self.model_sett["d"])]
        result = self.monte_carlo_integral(int_func, bounds)
        return result

    def c_n(self, n, yn, yn_prime):
        """
        Compute the squared Euclidean cost between yn and yn_prime at time step n.

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
            Squared Euclidean distance between yn and yn_prime.
        """
        return np.sum((yn - yn_prime) ** 2)

    def KL_0(self):
        """
        Compute the KL divergence for the initial marginal distribution of y_0.

        Returns
        -------
        float
            Estimated KL divergence for y_0.
        """

        def int_func(z):
            return np.log(
                self.param_model.q_0_marg1(z) / self.trans_kernel.m_0(z)
            ) * self.param_model.q_0_marg1(z)

        bound = tuple(map(int, self.model_sett["bound"].split(",")))
        bounds = [bound for _ in range(self.model_sett["d"])]
        result = self.monte_carlo_integral(int_func, bounds)
        return result

    def KL_0_prime(self):
        """
        Compute the KL divergence for the initial marginal distribution of y'_0.

        Returns
        -------
        float
            Estimated KL divergence for y'_0.
        """

        def int_func(z):
            return np.log(
                self.param_model.q_0_marg2(z) / self.trans_kernel.m_0_prime(z)
            ) * self.param_model.q_0_marg2(z)

        bound = tuple(map(int, self.model_sett["bound"].split(",")))
        bounds = [bound for _ in range(self.model_sett["d"])]
        result = self.monte_carlo_integral(int_func, bounds)
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

        def int_func(z):
            return np.log(
                self.param_model.q_n_marg1(n, z, ynm1, ynm1_prime)
                / self.trans_kernel.m_n(n, z, ynm1)
            ) * self.param_model.q_n_marg1(n, z, ynm1, ynm1_prime)

        bound = tuple(map(int, self.model_sett["bound"].split(",")))
        bounds = [bound for _ in range(self.model_sett["d"])]
        result = self.monte_carlo_integral(int_func, bounds)
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

        def int_func(z):
            return np.log(
                self.param_model.q_n_marg2(n, z, ynm1, ynm1_prime)
                / self.trans_kernel.m_n_prime(n, z, ynm1_prime)
            ) * self.param_model.q_n_marg2(n, z, ynm1, ynm1_prime)

        bound = tuple(map(int, self.model_sett["bound"].split(",")))
        bounds = [bound for _ in range(self.model_sett["d"])]
        result = self.monte_carlo_integral(int_func, bounds)
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
        Compute the cost-to-go at time step n, including KL terms and expected future costs.

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
            Cost-to-go at time step n.
        """
        if n == self.model_sett["N"]:
            return self.U_N(n, yn, yn_prime)
        beta = self.model_sett["beta"]
        B = self.model_sett["n_sim"]
        N = self.model_sett["N"]
        d = self.model_sett["d"]
        exp_ls = []
        for b in range(B):
            exp_inst = 0
            trajectory = np.zeros((N, 2, d))
            for i in range(N):
                if i == 0:
                    y0, y0_prime = self.param_model.get_ypair(n=0)
                    trajectory[i, 0, :] = y0
                    trajectory[i, 1, :] = y0_prime
                else:
                    y_prev, y_prev_prime = trajectory[n - 1]
                    y, y_prime = self.param_model.get_ypair(
                        n=i, ynm1=y_prev, ynm1_prime=y_prev_prime
                    )
                    trajectory[i, 0, :] = y
                    trajectory[i, 1, :] = y_prime

            for k in range(n + 1, N - 1):
                exp_inst += self.c_n(
                    k, yn=trajectory[k, 0, :], yn_prime=trajectory[k, 1, :]
                ) + beta * (
                    self.KL_n(
                        k + 1, ynm1=trajectory[k, 0, :], ynm1_prime=trajectory[k, 1, :]
                    )
                )
            exp_inst += self.c_n(
                N, yn=trajectory[N - 1, 0, :], yn_prime=trajectory[N - 1, 1, :]
            )
            exp_ls.append(exp_inst)
        exp = (1 / B) * sum(exp_ls)

        return (
            self.c_n(n, yn, yn_prime)
            + beta
            * (self.KL_n(n + 1, yn, yn_prime) + self.KL_n_prime(n + 1, yn, yn_prime))
            + exp
        )

    def jacobian_softmax_n(self, n):
        """
        Compute the Jacobian of the softmax function for the nth parameter vector.

        Parameters
        ----------
        n : int
            Index of the parameter vector.

        Returns
        -------
        np.ndarray
            Jacobian matrix of the softmax for theta[n].
        """
        return np.diag(self.param_model.theta[n]) - np.dot(
            self.param_model.theta[n], self.param_model.theta[n]
        )


class GaussianModel:
    def __init__(self, run_sett: dict):
        """
        Initialize the GaussianModel object.

        Parameters
        ----------
        run_sett : dict
            Dictionary containing model and run settings.
        """
        self.model_sett = run_sett["models"]["LinearMDP"]
        self.w = np.array(self.model_sett["init_w"])
        self.theta = self.batched_softmax(self.w)

    def batched_softmax(self, w):
        """
        Computes row-wise softmax over a (N, d) array of logits.

        Parameters
        ----------
        w : np.ndarray
            Input array of shape (N, d) or (d,).

        Returns
        -------
        np.ndarray
            Softmax output of shape (N, d) or (d,).
        """
        e = np.exp(w)
        if w.ndim == 1:
            return e / np.sum(e)
        else:
            return e / np.sum(e, axis=1, keepdims=True)

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
        k = self.theta.shape[1]
        d = self.model_sett["d"]
        if ynm1 is None and ynm1_prime is None:
            i = np.random.choice(k, p=self.theta[0])
            y0 = self.phi_0_objects(i)[1].rvs()
            y0_prime = self.phi_0_objects(i)[2].rvs()
            return y0, y0_prime
        else:
            i = np.random.choice(k, p=self.theta[n])
            yn = self.phi_n_objects(ynm1, ynm1_prime, i)[1].rvs()
            yn_prime = self.phi_n_objects(ynm1, ynm1_prime, i)[2].rvs()
            return yn, yn_prime

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
                self.phi_n_objects(ynm1, ynm1_prime, i)[0].pdf(yyprime)
                for i in range(self.model_sett["d"])
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
            [
                self.phi_n_objects(ynm1, ynm1_prime, i)[1].pdf(yn)
                for i in range(self.model_sett["d"])
            ]
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
                self.phi_n_objects(ynm1, ynm1_prime, i)[2].pdf(yn_prime)
                for i in range(self.model_sett["d"])
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
                self.phi_0_objects(i)[0].pdf(np.concatenate([y0, y0_prime]))
                for i in range(self.model_sett["d"])
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
        return np.array(
            [self.phi_0_objects(i)[1].pdf(y0) for i in range(self.model_sett["d"])]
        )

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
            [
                self.phi_0_objects(i)[2].pdf(y0_prime)
                for i in range(self.model_sett["d"])
            ]
        )

    def phi_n_objects(self, ynm1, ynm1_prime, i=None):
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
            Tuple of (joint, marginal y_n, marginal y'_n) multivariate normal distributions.
        """
        if i is None:
            i = 1
        d = self.model_sett["d"]
        A = 1.1 ** (i + 1) * np.eye(d)
        # B = 0.1*(i+1) * np.eye(d)
        B = np.zeros((d, d))
        # C = 0.1*(i+1) * np.eye(d)
        C = np.zeros((d, d))
        D = 1.1 ** (i + 1) * np.eye(d)

        mu_y = A @ ynm1 + B @ ynm1_prime
        mu_y_prime = C @ ynm1 + D @ ynm1_prime
        mu = np.concatenate([mu_y, mu_y_prime])

        Sigma = np.eye(2 * d)
        cov1 = np.array(Sigma[0:d, 0:d])
        cov2 = np.array(Sigma[d : 2 * d, d : 2 * d])

        return (
            multivariate_normal(mean=mu, cov=Sigma),  # type: ignore
            multivariate_normal(mean=mu_y, cov=cov1),  # type: ignore
            multivariate_normal(mean=mu_y_prime, cov=cov2),  # type: ignore
        )

    def phi_0_objects(self, i=None):
        """
        Return the joint and marginal multivariate normal distributions for (y_0, y'_0), y_0, and y'_0 for a given component.

        Parameters
        ----------
        i : int, optional
            Component index (default: 1).

        Returns
        -------
        tuple
            Tuple of (joint, marginal y_0, marginal y'_0) multivariate normal distributions.
        """
        if i is None:
            i = 1
        d = self.model_sett["d"]
        A = 0.5 ** (i + 1) * np.ones(d)
        B = 0.5 ** (i + 1) * np.ones(d)

        mu_y0 = A
        mu_y0_prime = B
        mu = np.concatenate([mu_y0, mu_y0_prime])

        Sigma = np.eye(2 * d)

        return (
            multivariate_normal(mean=mu, cov=Sigma),  # type: ignore
            multivariate_normal(mean=mu_y0, cov=Sigma[0:d, 0:d]),  # type: ignore
            multivariate_normal(mean=mu_y0_prime, cov=Sigma[d : 2 * d, d : 2 * d]),  # type: ignore
        )


class GaussianKernel:
    def __init__(self, run_sett: dict):
        """
        Initialize the GaussianKernel object.

        Parameters
        ----------
        run_sett : dict
            Dictionary containing model and run settings.
        """
        self.model_sett = run_sett["models"]["LinearMDP"]
        self.true_w = self.model_sett["true_w"]
        self.true_theta = self.batched_softmax(self.true_w)
        self.trajectory = self.get_trajectory()

    def batched_softmax(self, w):
        """
        Computes row-wise softmax over a (N, d) array of logits.

        Parameters
        ----------
        w : np.ndarray
            Input array of shape (N, d).

        Returns
        -------
        np.ndarray
            Softmax output of shape (N, d).
        """
        e = np.exp(w)
        return e / np.sum(e, axis=1, keepdims=True)

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
        d = self.model_sett["d"]
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
        d = self.model_sett["d"]
        return sum(self.true_theta[n][k] * components[k].pdf(value) for k in range(d))

    def m_0_objects(self):
        """
        Return the mixture components for y_0 and y'_0.

        Returns
        -------
        tuple
            Tuple of lists of multivariate normal components for y_0 and y'_0.
        """
        d = self.model_sett["d"]
        As = [0.5 ** (i + 1) * np.ones(d) for i in range(d)]
        Bs = [0.5 ** (i + 1) * np.ones(d) for i in range(d)]

        Sigma = np.eye(d)

        components_y0 = [multivariate_normal(mean=As[i], cov=Sigma) for i in range(d)]  # type: ignore
        components_y0_prime = [multivariate_normal(mean=Bs[i], cov=Sigma) for i in range(d)]  # type: ignore

        return components_y0, components_y0_prime

    def m_0(self, y0):
        """
        Compute the mixture PDF for y_0.

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
        Compute the mixture PDF for y'_0.

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
        Return the mixture components for y_n and/or y'_n at time n.

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
        d = self.model_sett["d"]
        As = [1.1 ** (i + 1) * np.eye(d) for i in range(d)]
        Ds = [1.1 ** (i + 1) * np.eye(d) for i in range(d)]
        Sigma = np.eye(d)

        if ynm1_prime is None and ynm1 is not None:
            components_yn = [multivariate_normal(mean=As[i] @ ynm1, cov=Sigma) for i in range(d)]  # type: ignore
            return components_yn
        elif ynm1 is None and ynm1_prime is not None:
            components_yn_prime = [multivariate_normal(mean=Ds[i] @ ynm1_prime, cov=Sigma) for i in range(d)]  # type: ignore
            return components_yn_prime
        elif ynm1 is not None and ynm1_prime is not None:
            components_yn = [multivariate_normal(mean=As[i] @ ynm1, cov=Sigma) for i in range(d)]  # type: ignore
            components_yn_prime = [multivariate_normal(mean=Ds[i] @ ynm1_prime, cov=Sigma) for i in range(d)]  # type: ignore
            return components_yn, components_yn_prime

    def m_n(self, n, yn, ynm1):
        """
        Compute the mixture PDF for y_n at time n.

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
        Compute the mixture PDF for y'_n at time n.

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

    def get_trajectory(self):
        """
        Create the y and y' sequences according to the assumed Markovian transition kernels, assuming Gaussianity.

        Returns
        -------
        np.ndarray
            Trajectory array of shape (N, 2, d) containing y and y' for all time steps.
        """
        N = self.model_sett["N"]
        d = self.model_sett["d"]
        trajectory = np.zeros((N, 2, d))

        # Initial values y_0 and y_0' ~ N(0, I_d)
        y0 = self.MVNMixture_rvs(self.m_0_objects()[0], n=0)
        y0_prime = self.MVNMixture_rvs(self.m_0_objects()[1], n=0)

        trajectory[0, 0, :] = y0
        trajectory[0, 1, :] = y0_prime

        # Transition: y_n ~ N(y_{n-1}, I_d), and similarly for y'
        for n in range(1, N):
            yn = self.MVNMixture_rvs(
                self.m_n_objects(n, ynm1=trajectory[n - 1, 0, :]), n=n
            )
            yn_prime = self.MVNMixture_rvs(
                self.m_n_objects(n, ynm1_prime=trajectory[n - 1, 1, :]), n=n
            )
            trajectory[n, 0, :] = yn
            trajectory[n, 1, :] = yn_prime

        return trajectory
