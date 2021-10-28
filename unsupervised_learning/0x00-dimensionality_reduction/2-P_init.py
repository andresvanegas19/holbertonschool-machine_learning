#!/usr/bin/env python3
""" Do the Dimensionality Reduction """
import numpy as np


def P_init(X, perplexity):
    """
    initializes all variables required to calculate the P affinities in t-SNE

    Args:
        X= is a numpy.ndarray of shape (n, d) where:
            n is the number of data points
            d is the number of dimensions in each point
            all dimensions have a mean of 0 across all data points
        perplexity is the perplexity that all Gaussian
        distributions should have

    Returns:
        the weights matrix, W, that maintains var
        fraction of X‘s original variance
    """

    a, _ = X.shape

    tot = np.sum(
        np.square(X),
        axis=1
    )

    # Compute P-values
    Di = np.add(
        np.add(-2 * np.matmul(X, X.T), tot).T, tot
    )

    # compute P-values
    np.fill_diagonal(Di, 0)

    return Di, np.zeros((a, a)), np.ones((a, 1)), np.log2(perplexity)
