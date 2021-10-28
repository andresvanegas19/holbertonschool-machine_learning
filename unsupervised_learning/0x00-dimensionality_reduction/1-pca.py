#!/usr/bin/env python3
""" Do the Dimensionality Reduction """
import numpy as np


def pca(X, ndim):
    """
    performs PCA on a dataset:

    Args:
        X= is a numpy.ndarray of shape (n, d) where:
            n is the number of data points
            d is the number of dimensions in each point
            all dimensions have a mean of 0 across all data points
        ndim is the new dimensionality of the transformed X

    W is a numpy.ndarray of shape (d, nd) where
    nd is the new dimensionality of the transformed X

    Returns:
        the weights matrix, W, that maintains var
        fraction of X‘s original variance
    """
    m = X - np.mean(X, axis=0)
    return np.dot(
        m,
        np.linalg.svd(m)[2].T[:, :ndim]
    )
