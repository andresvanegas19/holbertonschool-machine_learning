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
    t, m, _ = X.shape
    # output
    L, _, h = h_0.shape
    H = np.zeros((t + 1, L, m, h))
    H[0, :, :, :] = h_0
    Y = []

    for step in range(t):
        for layer in range(L):
            # Calculate the output of the layer and not pass
            if layer == 0:
                h_next, y = rnn_cells[layer].forward(H[step, layer], X[step])
            else:
                h_next, y = rnn_cells[layer].forward(H[step, layer], h_next)
            H[step + 1, layer] = h_next
        Y.append(y)
    Y = np.array(Y)

    return H, Y
