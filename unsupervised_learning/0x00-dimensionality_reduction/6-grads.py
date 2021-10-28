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

    # dY is the gradient of Y
    dY = np.zeros((a, dim))
    # b is the gradient of the Q affinities
    grandB = np.expand_dims(((P - affinQ) * num).T, axis=2)

    for i in range(a):
        dY[i, :] = np.sum((grandB[i, :] * Y[i, :] - Y), 0)

    return dY, affinQ
