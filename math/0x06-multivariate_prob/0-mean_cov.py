#!/usr/bin/env python3
""" covariance mean calculating """

import numpy as np


def mean_cov(X):
    """
    calculates the mean and covariance of a data set

    Args:
        X: is a numpy.ndarray of shape (n, d) containing the data set

    Raises:
        TypeError: with the message X must be a 2D numpy.ndarray
        ValueError: n is less than 2, with the message X must
        contain multiple data points

    Returns:
        cov:  is a numpy.ndarray of shape (d, d)
        containing the covariance matrix of the data set

        mean: is a numpy.ndarray of shape (1, d) containing
        the mean of the data set
    """

    if len(X.shape) != 2 or type(X) != np.ndarray:
        raise TypeError("X must be a 2D numpy.ndarray")

    if X.shape[0] < 2:
        raise ValueError("X must contain multiple data points")


    data_points, dimension = X.shape
    mean = np.mean(X, axis=0).reshape(1, dimension)
    tot = X - mean
    cov = np.dot(tot.T, tot) / (data_points - 1)
    return mean, cov
