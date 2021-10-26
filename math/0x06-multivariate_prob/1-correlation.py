#!/usr/bin/env python3
""" Correlation of the matrix """
import numpy as np


def correlation(C):
    """
    Multivariate Normal distribution

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

    if type(C) is not np.ndarray:
        raise TypeError("C must be a numpy.ndarray")

    if C.shape[0] != C.shape[1] or len(C.shape) != 2:
        raise ValueError("C must be a 2D square matrix")

    std = np.sqrt(np.diag(C))
    #  (v1, v2, ... vn)
    outer_product = np.outer(std, std)

    return C / outer_product
