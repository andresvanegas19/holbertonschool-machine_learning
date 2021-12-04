#!/usr/bin/env python3
""" RNNs """
import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    performs forward propagation for a simple RNN

    Args:
        rnn_cell: is class instace (RRNC)  for the forward propagation use
        X: Should, given as a numpy.ndarray of shape (t, m, i)
            - t is the maximum number of time steps
            - m is the batch size
            - i is the dimensionality of the data
        h_0: is the initial hidden state

    Returns:
        - first: all of the hidden states
        - second: all of the outputs
    """
    h_next = h_0.copy()
    H = [h_next]

    all_inp_Y = []

    for t in range(X.shape[0]):
        h_next, y = rnn_cell.forward(h_next, X[t])
        H.append(h_next)
        all_inp_Y.append(y)

    return np.array(H), np.array(all_inp_Y)
