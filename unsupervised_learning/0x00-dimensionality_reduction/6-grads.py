#!/usr/bin/env python3
""" Do the Dimensionality Reduction """
import numpy as np
Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """
    this function calculates the gradients used for
    the gradient descent update during the training
    of the Q affinities

    Returns:
        dY is a numpy.ndarray of shape (n, ndim) containing the gradients of Y
        Q is a numpy.ndarray of shape (n, n) containing the Q affinities of Y
    """

    a, dim = Y.shape
    # q is the Q affinities of Y
    # num is the number of points in Y
    affinQ, num = Q_affinities(Y)

    # dY is the gradient of Y filled by zeros
    dY = np.zeros((a, dim))
    # b is the gradient of the Q affinities
    PQ = P - affinQ

    for i in range(a):
        # here we calculate the gradient of Y for
        # each point i in Y and add it dY
        dY[i, :] = np.sum(
            np.tile(PQ[:, i] * num[:, i], (dim, 1)).T * (Y[i, :] - Y),
            0
        )

    return dY, affinQ
