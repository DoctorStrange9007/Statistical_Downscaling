import numpy as np
from scipy.integrate import nquad
from scipy.stats import multivariate_normal


class LinearMDP:
    """"""

    def __init__(self, run_sett: dict, param_model, trans_kernel):

        self.model_sett = run_sett["models"]["LinearMDP"]
        self.param_model = param_model
        self.trans_kernel = trans_kernel
        self.beta = self.model_sett["beta"]

    def sgd(self, lr):
        N = self.model_sett["N"]
        T = self.model_sett["T"]
        for t in range(T):
            for n in range(N):
                if n == 0:
                    self.sgd_0(lr)
                else:
                    self.sgd_n(lr, n)

    def sgd_0(self, lr):
        grad = self.compute_gradient_0(self.param_models.theta[0])
        self.param_model.theta[0] -= lr * grad

    def sgd_n(self, lr, n):
        grads = self.compute_gradient_n(self.param_model.theta[n], n)
        for n, grad in enumerate(grads):
            self.param_model.theta[n] -= lr * grad

    def compute_gradient_0(self):
        grads = []
        beta = self.model_sett["beta"]
        B = 100
        for b in range(B):
            y0, y0_prime = self.param_models[0].get_y_pair()
            grad_b += U_n(y0, y0_prime) * (
                self.param_model.phi_0(y0, y0_prime)
                / np.dot(
                    self.param_model.phi_0(y0, y0_prime), self.param_model.theta[0]
                )
            )
            grads.append(grad_b)

        grad = (1 / B) * sum(grads) + beta * (
            self.gradient_KL_0() + self.gradient_KL_0_prime()
        )

        return grad

    def compute_gradient_n(self, n):
        grads = []
        B = 100
        beta = self.model_sett["beta"]

        for b in range(B):
            for i in range(n):
                if n == 0:
                    y0, y0_prime = self.param_model.get_y_pair()
                    trajectory = [(y0, y0_prime)]
                else:
                    y_prev, y_prev_prime = trajectory[n - 1]
                    y, y_prime = self.param_model.get_y_pair(
                        ynmin1=y_prev, ynm1_prime=y_prev_prime
                    )
                    trajectory.append((y, y_prime))

            grad_b += U_n(y, y_prime) * (
                self.param_model.phi_n(y, y_prime, y_prev, y_prev_prime)
                / np.dot(
                    self.param_model.phi_n(y, y_prime, y_prev, y_prev_prime),
                    self.param_model.theta[n],
                )
            )
            grad_b += beta * (
                self.gradient_KL_n_prime(y, y_prime) + self.gradient_KL_n(y, y_prime)
            )

            grads.append(grad_b)

        return (1 / B) * sum(grads)

    def gradient_KL_n_prime(self, ynmin1, ynmin1_prime):
        def int_func(z):
            np.log(
                np.dot(self.phi_marg2(z, ynmin1, ynmin1_prime), self.theta[n])
                / self.m_n_prime(z, ynmin1_prime)
            ) * self.phi_n_marg2(z, ynmin1, ynmin1_prime)

        result, error = nquad(int_func, [[-np.inf, np.inf]] * len(self.model_sett["d"]))

        return result

    def gradient_KL_n(self, n, ynmin1, ynmin1_prime):
        def int_func(z):
            np.log(
                np.dot(self.phi_n_marg1(z, ynmin1, ynmin1_prime), self.theta[n])
                / self.m_n(z, ynmin1)
            ) * self.phi_n_marg1(z, ynmin1, ynmin1_prime)

        result, error = nquad(int_func, [[-np.inf, np.inf]] * len(self.model_sett["d"]))

        return result

    def gradient_KL_0_prime(self):
        def int_func(z):
            np.log(
                np.dot(self.phi_0_marg2(z), self.theta[0]) / self.m_0_prime(z)
            ) * self.phi_0_marg2(z)

        result, error = nquad(int_func, [[-np.inf, np.inf]] * len(self.model_sett["d"]))

        return result

    def gradient_KL_0(self):
        def int_func(z):
            np.log(
                np.dot(self.phi_0_marg1(z), self.theta[0]) / self.m_0(z)
            ) * self.phi_0_marg1(z)

        result, error = nquad(int_func, [[-np.inf, np.inf]] * len(self.model_sett["d"]))

        return result


class GaussianModel(LinearMDP):
    def __init__(self, model_sett: dict, theta_init: np.array):
        super().__init__(model_sett)

        self.theta = theta_init

    def get_ypair(self, n=None, ynm1=None, ynm1_prime=None):
        k = len(self.theta)
        if ynm1 == None and ynm1_prime == None:
            i = np.random.choice(k, p=self.theta[0])
            return self.phi_0_object()[1].rvs(), self.phi_0_object()[2].rvs()
        else:
            i = np.random.choice(k, p=self.theta[n])
            return (
                self.phi_n_object(ynm1, ynm1_prime)[1].rvs(),
                self.phi_n_object(ynm1, ynm1_prime)[2].rvs(),
            )

    def q_n_marg2(self, n, yn_prime, ynm1, ynm1_prime):

        return np.dot(self.phi_n_marg2(yn_prime, ynm1, ynm1_prime), self.theta[n])

    def q_n_marg1(self, yn, ynm1, ynm1_prime):

        return np.dot(self.phi_n_marg1(yn, ynm1, ynm1_prime), self.theta[n])

    def q_n(self, yn, yn_prime, ynm1, ynm1_prime):

        return np.dot(self.phi_n(yn, yn_prime, ynm1, ynm1_prime), self.theta[n])

    def q_0(self, y0, y0_prime):

        return np.dot(self.phi_0(y0, y0_prime), self.theta[0])

    def phi_n(self, yn, yn_prime, ynm1, ynm1_prime):

        yyprime = np.array([yn, yn_prime])

        return self.phi_n_objects(ynm1, ynm1_prime)[0].pdf(yyprime)

    def phi_n_marg1(self, yn, ynm1, ynm1_prime):

        return self.phi_n_objects(ynm1, ynm1_prime)[1].pdf(yn)

    def phi_n_marg2(self, yn_prime, ynm1, ynm1_prime):

        return self.phi_n_objects(ynm1, ynm1_prime)[2].pdf(yn_prime)

    def phi_0(self, y0, y0_prime):

        return self.phi_0_objects[0].pdf(y0, y0_prime)

    def phi_0_marg1(self, y0):

        return self.phi_0_objects[1].pdf(y0)

    def phi_0_marg1(self, y0_prime):

        return self.phi_0_objects[2].pdf(y0_prime)

    def phi_n_objects(self, ynm1, ynm1_prime):
        d = self.model_sett["d"]
        A = np.eye(d)
        B = 0.5 * np.eye(d)
        C = 0.9 * np.eye(d)
        D = np.eye(d)

        mu_y = A @ ynm1 + B @ ynm1_prime
        mu_y_prime = C @ ynm1 + D @ ynm1_prime
        mu = np.concatenate([mu_y, mu_y_prime])

        Sigma = np.eye(2 * d)

        return (
            multivariate_normal(mean=mu, cov=Sigma),
            multivariate_normal(mean=mu_y, cov=Sigma[0:d, 0:d]),
            multivariate_normal(mean=mu_y_prime, cov=Sigma[d : 2 * d, d : 2 * d]),
        )

    def phi_0_objects(self):
        d = self.model_sett["d"]

        mu_y0 = np.array([0] * d)
        mu_y0_prime = np.array([0] * d)
        mu = np.concatenate([mu_y0, mu_y0_prime])

        Sigma = np.eye(d)

        return (
            multivariate_normal(mean=mu, cov=Sigma),
            multivariate_normal(mean=mu_y0, cov=Sigma[0:d, 0:d]),
            multivariate_normal(mean=mu_y0_prime, cov=Sigma[d : 2 * d, d : 2 * d]),
        )


class GaussianKernel(LinearMDP):
    def __init__(self, model_sett: dict):
        super().__init__(model_sett)

        self.trajectory = self.get_trajectory()

    def m_0_objects(self):
        "for y and y', respectively"
        d = self.model_sett["d"]
        mu = np.zeros(d)
        Sigma = np.eye(d)

        return multivariate_normal(mean=mu, cov=Sigma), multivariate_normal(
            mean=mu, cov=Sigma
        )

    def m_0(self, y0):

        return self.m_0_objects()[0].pdf(y0)

    def m_0_prime(self, y0_prime):

        return self.m_0_objects()[1].pdf(y0_prime)

    def m_n_objects(self, ynmin1=None, ynmin1prime=None):
        d = self.model_sett["d"]
        Sigma = np.eye(d)

        if ynmin1prime == None:
            return multivariate_normal(mean=ynmin1, cov=Sigma)
        elif ynmin1 == None:
            return multivariate_normal(mean=ynmin1prime, cov=Sigma)
        else:
            return multivariate_normal(mean=ynmin1, cov=Sigma), multivariate_normal(
                mean=ynmin1prime, cov=Sigma
            )

    def m_n(self, yn, ynmin1):

        return self.m_n_objects(ynmin1=ynmin1).pdf(yn)

    def m_n_prime(self, ynprime, ynmin1prime):

        return self.m_n_objects(ynmin1prime=ynmin1prime).pdf(ynprime)

    def theta(self, w):
        np.exp(w) / np.sum(np.exp(w))

    def get_trajectory(self):
        """Create the y and y' sequences accoring to the assumed markovian transition kernels. Assuming Gaussianity."""
        T = self.model_sett["T"]
        d = self.model_sett["d"]

        # Initial values y_0 and y_0' ~ N(0, I_d)
        y0 = self.m_0_objects()[0].rvs()
        y0_prime = self.m_0_objects()[1].rvs()

        trajectory = np.array([y0, y0_prime])

        # Transition: y_n ~ N(y_{n-1}, I_d), and similarly for y'
        for n in range(T):
            yn = self.m_n_objects(ynmin1=trajectory[2 * n, :]).rvs()
            yn_prime = self.m_n_objects(ynmin1_prime=trajectory[2 * n + 1, :]).rvs()
            trajectory.append([yn, yn_prime])

        return trajectory
