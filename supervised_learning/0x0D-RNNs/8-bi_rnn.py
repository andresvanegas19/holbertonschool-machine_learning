#!/usr/bin/env python3
""" RNNs """
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    performs forward propagation for a bidirectional RNN

    Args:
        bi_cell(BidirectinalCell): will be used for the forward propagation
        X: is the data to be used
            - t is the maximum number of time steps
            - m is the batch size
            - i is the dimensionality of the data
        h_0: is the initial hidden state in the forward direction
            - h is the dimensionality of the hidden state
        h_t: is the initial hidden state in the backward direction

    Returns:
        - H: concatenated hidden states
        - Y: containing all of the outputs
    """
    T, m, _ = X.shape
    m, h = h_0.shape
    H = np.zeros((T, m, h * 2))
    h_prev = np.zeros((T, m, h))
    h_next = np.zeros((T, m, h))
    Y = []
    x_f = h_0
    # x_b is
    x_b = h_t

    for step in range(T):
        h_next[step] = bi_cell.forward(x_f, X[step])
        h_prev[T - step - 1] = bi_cell.backward(x_b, X[T - step - 1])
        x_b = h_prev[T - step - 1]
        x_f = h_next[step]

    H = np.concatenate((h_next, h_prev), axis=-1)
    Y = bi_cell.output(H)

    return H, np.array(Y)
