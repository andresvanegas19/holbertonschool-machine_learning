#!/usr/bin/env python3
""" Class BayesianOptimization"""
import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """ Class that performs Bayesin """

    def __init__(
            self,
            f,
            X_init,
            Y_init,
            bounds,
            ac_samples,
            l=1,
            sigma_f=1,
            xsi=0.01,
            minimize=True
    ):
        """
        performs Bayesian optimization on a noiseless 1D Gaussian process

        Args:
            f: is the black-box function to be optimized
            X_init: is a numpy.ndarray of shape (t, 1) representing the inputs
                already sampled with the black-box function
            Y_init: is a numpy.ndarray of shape (t, 1) representing the outputs
                of the black-box function for each input in X_init
            t: is the number of initial samples
            bounds: is a tuple of (min, max) representing the bounds of the
                space in which to look for the optimal point
            ac_samples: is the number of samples that should be analyzed
                during acquisition
            l: is the length parameter for the kernel
            sigma_f: is the standard deviation given to the output of
                the black-box function
            xsi: is the exploration-exploitation factor for acquisition
            minimize: is a bool determining whether optimization should be
                performed for minimization (True) or maximization (False)
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)

        min_v, max_v = bounds

        self.X_s = np.linspace(min_v, max_v, ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        Calculates the next best sample location

        Returns:
            - X_next is a numpy.ndarray of shape (1,) representing the next
                best sample point
            - EI is a numpy.ndarray of shape (ac_samples,) containing
                the expected improvement of each potential sample
        """

        mu, sigmna = self.gp.predict(self.X_s)
        # z_arr_sam array sample
        z_arr_sam = np.zeros(sigmna.shape[0])

        if (self.minimize):
            mu_s_opt = np.min(self.gp.Y)
            improve = mu_s_opt - mu - self.xsi
        else:
            mu_s_opt = np.max(self.gp.Y)
            improve = mu - mu_s_opt - self.xsi

        for i in range(sigmna.shape[0]):
            if sigmna[i] > 0:
                z_arr_sam[i] = improve[i] / sigmna[i]
            else:
                z_arr_sam[i] = 0
            EI = improve * norm.cdf(z_arr_sam) + sigmna * norm.pdf(z_arr_sam)

        return self.X_s[np.argmax(EI)], EI

    def optimize(self, iterations=100):
        """
        Optimizes the black-box function

        args:
            iterations: is the number of iterations to perform

        Returns:
            X_opt is a numpy.ndarray of shape (1,)
                representing the optimal point
            Y_opt is a numpy.ndarray of shape (1,)
                representing the optimal function value
        """
        for _ in range(0, iterations):
            X_next, _ = self.acquisition()
            if X_next in self.gp.X:
                break

            self.gp.update(X_next, self.f(X_next))

        if self.minimize is True:
            idx = np.argmin(self.gp.Y)
        else:
            idx = np.argmax(self.gp.Y)

        return self.gp.X[idx], self.gp.Y[idx]
