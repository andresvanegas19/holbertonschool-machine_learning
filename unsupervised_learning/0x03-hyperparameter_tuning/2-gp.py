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

        args:
            X1: is a numpy.ndarray of shape (n, 1)
            X2: is a numpy.ndarray of shape (m, 1)

        Returns:
            it will retrun a numpy.ndarray of shape (n, m)
        """

        sqdist = np.sum(X1 ** 2, 1).reshape(-1, 1) + \
            np.sum(X2 ** 2, 1) - (2 * np.dot(X1, X2.T))

        return self.sigma_f ** 2 * np.exp(-0.5 / self.l ** 2 * sqdist)

    def predict(self, X_s):
        """
        Method that predicts the mean and standard deviation of points in
        a Gaussian process

        args:
            X_s: is a numpy.ndarray of shape (s, 1) containing
            all of the points

        Returns:
            mu: is a numpy.ndarray of shape (s, 1) containing the mean
        """
        convs_m = self.kernel(self.X, X_s)
        kinv = np.linalg.inv(self.K)

        mu = np.dot(convs_m.T, kinv).dot(self.Y)
        mu = mu.reshape((X_s.shape[0]))

        # 2,  is sigma
        return mu, np.diag(
            self.kernel(X_s, X_s) -
            np.dot(convs_m.T, kinv).dot(convs_m)
        )

    def update(self, X_new, Y_new):
        """
        updates a Gaussian Process, i.e., the mean and standard deviation of
        the points in X_new and Y_new

        args:
            X_new: is a numpy.ndarray of shape (1,) that represents the new
            Y_new: is a numpy.ndarray of shape (1,) that represents the new ..

        Returns:
            None
        """

        self.X = np.append(self.X, X_new).reshape(-1, 1)
        self.Y = np.append(self.Y, Y_new).reshape(-1, 1)
        # K = self.kernel(self.X, self.X)
        self.K = self.kernel(self.X, self.X)
