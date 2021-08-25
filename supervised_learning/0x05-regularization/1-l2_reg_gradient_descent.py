#!/usr/bin/env python3
""" Module for regularization method and technicths"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    updates the weights and biases of a neural network using gradient descent
    with L2 regularization:

    Y is a one-hot numpy.ndarray of shape (classes, m) that contains the
    correct labels for the data
        - classes is the number of classes
        - m is the number of data points
    weights is a dictionary of the weights and biases of the neural network
    cache is a dictionary of the outputs of each layer of the neural network
    alpha is the learning rate
    lambtha is the L2 regularization parameter
    L is the number of layers of the network

    The weights and biases of the network should be updated in place
    The neural network uses tanh activations on each layer except
    the last, which uses a softmax activation
    """

    d_z = cache['A' + str(L)] - Y

    for i in range(L, 0, -1):

        cost_L = (lambtha * (1 / Y.shape[1])) * weights['W{}'.format(i)]
        dW = (
            (1 / Y.shape[1]) * np.matmul(d_z, cache['A' + str(i - 1)].T)
        ) + cost_L

        d_b = (1 / Y.shape[1]) * np.sum(d_z, axis=1, keepdims=True)

        d_z = np.matmul(weights['W' + str(i)].T, d_z)

        A = cache['A' + str(i - 1)]

        d_z *= (1 - np.power(A, 2))

        weights['W' + str(i)] -= (alpha * dW)
        weights['b' + str(i)] -= (alpha * d_b)
