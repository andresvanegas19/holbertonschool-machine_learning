#!/usr/bin/env python3
""" Module for regularization method and technicths"""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    updates the weights of a neural network with Dropout regularization
    using gradient descent:

    Y is a one-hot numpy.ndarray of shape (classes, m) that contains
    the correct labels for the data
        - classes is the number of classes
        - m is the number of data points
    weights is a dictionary of the weights and biases of the neural network
    cache is a dictionary of the outputs and dropout masks of
    each layer of the neural network
    alpha is the learning rate
    keep_prob is the probability that a node will be kept
    L is the number of layers of the network
    """

    d_Z = cache['A' + str(L)] - Y

    m1 = (1 / Y.shape[1])

    # pagination
    for i in range(L, 0, -1):
        d_W = m1 * np.matmul(d_Z, cache['A' + str(i - 1)].T)
        d_b = m1 * np.sum(d_Z, axis=1, keepdims=True)

        d_Z = np.matmul(weights['W' + str(i)].T, d_Z)

        A = cache['A' + str(i - 1)]

        if i > 1:
            d_Z *= (1 - np.power(A, 2)) * (cache['D' + str(i - 1)] / keep_prob)

        weights['b' + str(i)] -= (alpha * d_b)
        weights['W' + str(i)] -= (alpha * d_W)
