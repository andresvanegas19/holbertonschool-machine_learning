#!/usr/bin/env python3
""" RNNs """
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    performs forward propagation for a deep RNN

    Args:
        rnn_cells: is a list of RNNCell instances of length l that will be used
                    for the forward propagation
        X: is the data to be used, given as a numpy.ndarray of shape (t, m, i)
            - t is the maximum number of time steps
            - m is the batch size
            - i is the dimensionality of the data
        h_0: is the initial hidden state

    Returns:
        - H: containing all of the hidden states
        - Y: containing all of the outputs
    """
    h_pnex = h_0
    L, m, i = h_0.shape

    # Initialize H and Y
    H = np.zeros((X.shape[0] + 1, L, m, i))
    H[0] = h_0
    # Y is a matrix of shape (t, m, i) containing the outputs
    Y = []

    for i_s in range(X.shape[0]):
        h_prev = X[i_s]
        for layer in range(L):
            # here we are using the previous hidden state as the input
            h_pnex, y = rnn_cells[layer].forward(H[i_s, layer], h_prev)
            h_prev = h_pnex
            # store the hidden state
            H[i_s + 1, L, :, :] = h_pnex
        Y.append(y)

    return H, Y
