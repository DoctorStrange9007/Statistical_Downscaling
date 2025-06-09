import numpy as np
from scipy.integrate import nquad
from scipy.stats import multivariate_normal


class Model:
    """Base class for all financial models.

    Provides common functionality and structure for derived model classes.

    Parameters
    ----------
    run_sett : dict
        Dictionary containing all runtime settings and parameters
    clustered_dfs_sets : list
        List of dictionaries of DataFrames containing clustered financial data
    """

    def __init__(self, run_sett: dict):
        self._run_sett = run_sett


class LinearMDP(Model):
    """"""

    def __init__(self, run_sett: dict):
        super().__init__(run_sett)

        self.result = self.get_result()
        self.model_sett = self._run_sett["models"]["LinearMDP"]
        self.y_0_vec = np.random.randn(2, 10)
        self.beta = self.model_sett["beta"]

    def gradient_J_n(self):
        # \mathbb E [...]
        return self.beta * (
            self.gradient_KL_n + self.gradient_KL_n_prime
        )  # + self.cost(self.y_0_vec)/np.dot(self.phi(self.y_n_vec))*self.phi(self.y_n_vec)

    def gradient_J_0(self):
        return self.beta * (
            self.gradient_KL_0 + self.gradient_KL_n
        )  # +\mathbb E [ (self.cost(self.y_0_vec)/np.dot(self.phi_0())*self.phi_0() ]

    def gradient_KL_n_prime(self):
        def int_func(z):
            np.log(
                np.dot(self.phi_marginal_2(z, self.y_nmin1_vec), self.theta_n)
                / self.m_n_prime(z, self.y_nmin1_prime)
            ) * self.phi_marginal_2(z, self.y_min1_vec)

        result, error = nquad(int_func, [[-np.inf, np.inf]] * len(self.y))

        return result

    def gradient_KL_n(self):
        def int_func(z):
            np.log(
                np.dot(self.phi_marginal_1(z, self.y_nmin1_vec), self.theta_n)
                / self.m_n(z, self.y_nmin1)
            ) * self.phi_marginal_1(z, self.y_min1_vec)

        result, error = nquad(int_func, [[-np.inf, np.inf]] * len(self.y))

        return result

    def gradient_KL_0_prime(self):
        def int_func(z):
            np.log(
                np.dot(self.phi_0_marginal_2(z), self.theta_0) / self.m_0_prime(z)
            ) * self.phi_0_marginal_2(z)

        result, error = nquad(int_func, [[-np.inf, np.inf]] * len(self.y))

        return result

    def gradient_KL_0(self):
        def int_func(z):
            np.log(
                np.dot(self.phi_0_marginal_1(z), self.theta_0) / self.m_0(z)
            ) * self.phi_0_marginal_1(z)

        result, error = nquad(int_func, [[-np.inf, np.inf]] * len(self.y))

        return result

    def m_n_prime(self, y_n_prime, y_nmin1_prime):
        if self.model_sett["m"] == "normal":
            return multivariate_normal.pdf(
                y_n_prime, mean=y_nmin1_prime, cpv=np.eye(len(y_nmin1_prime))
            )

    def m_n(self, y_n, y_nmin1):
        if self.model_sett["m"] == "normal":
            return multivariate_normal.pdf(y_n, mean=y_nmin1, cpv=np.eye(len(y_nmin1)))

    def m_0(self):
        if self.model_sett["m"] == "normal":
            return multivariate_normal.pdf(
                self.y_0_vec[:, 0], mean=0, cpv=np.eye(len(self.y_0_vec[:, 0]))
            )

    def m_0_prime(self):
        if self.model_sett["m"] == "normal":
            return multivariate_normal.pdf(
                self.y_0_vec[:, 1], mean=0, cpv=np.eye(len(self.y_0_vec[:, 1]))
            )

    def q_n_marginal_2(self):
        pass

    def q_n_marginal_1(self):
        pass

    def q_n(self):
        pass

    def q_0(self):
        pass

    def phi_0_marginal_2(self):
        pass

    def phi_marginal_1(self):
        pass

    def phi_marginal_2(self):
        pass

    def phi_0_marginal_1(self):
        pass

    def phi(self, y_vec):
        if self.model_sett["phi_type"] == "linear":
            return np.sum(y_vec, axis=0)

    def phi_0(self):
        if self.model_sett["phi_type"] == "linear":
            return np.sum(self.y_0_vec, axis=0)
