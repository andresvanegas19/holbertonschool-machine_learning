#!/usr/bin/env python3
""" Module for regularization method and technicths"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    conducts forward propagation using Dropout:

    X is a numpy.ndarray of shape (nx, m) containing the
    input data for the network
        - nx is the number of input features
        - m is the number of data points
    weights is a dictionary of the weights and biases of the neural network
    L the number of layers in the network
    keep_prob is the probability that a node will be kept
    All layers except the last should use the tanh activation function
    The last layer should use the softmax activation function

    Returns: a dictionary containing the outputs of each layer and the
    dropout mask used on each layer (see example for format)
    """

    cache = {"A0": X}

    for i in range(L):
        z = np.matmul(
            weights["W" + str(i + 1)],
            cache["A" + str(i)]
        ) + weights["b" + str(i + 1)]

        drop = np.random.binomial(1, keep_prob, size=z.shape)

        if i == L - 1:
            cache["A" + str(i + 1)] = np.exp(z) / \
                np.sum(np.exp(z), axis=0, keepdims=True)

            return cache

        cache["D" + str(i + 1)] = drop
        cache["A" + str(i + 1)] = \
            np.tanh(z) * cache["D" + str(i + 1)] / keep_prob

    return cache
