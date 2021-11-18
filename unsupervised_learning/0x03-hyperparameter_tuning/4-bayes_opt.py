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

        mu, sigma = self.gp.predict(self.X_s)
        # mu_sample = self.p.predict(X_sample)

        # sigma = sigma.reshape(-1, 1)

        # sigm, _ = self.gp.predict(self.gp.X)
        # next_fs, gc_var = self.gp.predict(self.X_s)
        # opt = np.min(sigm)
        # improves = opt - next_fs - self.xsi
        # Needed for noise-based model,
        # otherwise use np.max(Y_sample).
        # See also section 2.4 in [1]
        if self.minimize is True:
            samp_y = np.min(self.gp.Y)
            improves = samp_y - mu - self.xsi
        # # Compute EI in negative space
        # if not self.minimize:
        #     improve = -improves
        else:
            samp_y = np.max(self.gp.Y)
            improves = mu - samp_y - self.xsi

        with np.errstate(divide='warn'):

            Z = improves / sigma
            ei = improves * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        # Z = improves / gc_var
        # eis = improves * norm.cdf(Z) + gc_var * norm.pdf(Z)

        return self.X_s[np.argmax(ei)], ei
