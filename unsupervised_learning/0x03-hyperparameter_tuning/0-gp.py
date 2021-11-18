#!/usr/bin/env python3
""" Hyperparameter Tuning - Gaussian Process """

import numpy as np


class GaussianProcess():
    """
    Compute the mean of the predictive distribution of the Gaussian Process
    """

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        init function for the Gaussian Process
        """
        self.X = X_init
        self.Y = Y_init
        self.sigma_f = sigma_f
        self.l = l
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """
        This function computes a covariance kernel matrix between two matrices
        and returns it
        """

        sqdist = np.sum(X1 ** 2, 1).reshape(-1, 1) + \
            np.sum(X2 ** 2, 1) - (2 * np.dot(X1, X2.T))

        return self.sigma_f ** 2 * np.exp(-0.5 / self.l ** 2 * sqdist)
