#!/usr/bin/env python3
"""  performs PCA  """
import numpy as np


def pca(X, var=0.95):
    """
    performs PCA on a dataset:

    Args:
        X= is a numpy.ndarray of shape (n, d) where:
            n is the number of data points
            d is the number of dimensions in each point
            all dimensions have a mean of 0 across all data points
        var = is the fraction of the variance that the
        PCA transformation should maintain

    W is a numpy.ndarray of shape (d, nd) where
    nd is the new dimensionality of the transformed X

    Returns:
        the weights matrix, W, that maintains var
        fraction of X‘s original variance
    """
    _, s, vh = np.linalg.svd(X)

    cs = np.cumsum(s)

    dim = [i for i in range(len(s)) if cs[i] / cs[-1] >= var]

    return vh.T[:, :dim[0] + 1]
