#!/usr/bin/env python3
""" Do clustering scikit-learn """

import numpy as np


def initialize(X, k):
    """
    This function initializes the centroids for K-means

    Args:
        X: is a numpy.ndarray of shape (n, d) containing the dataset that
        we want to cluster
        k: is a positive integer containing the number of clusters

    Returns:
        [type]: [description]
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None

    if type(k) != int or k <= 0 or k >= X.shape[0]:
        return None

    # initialize the centroids
    _, d = X.shape

    return np.random.uniform(
        np.amin(X, axis=0),
        np.amax(X, axis=0),
        (k, d)
    )
