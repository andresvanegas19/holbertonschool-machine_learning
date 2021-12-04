#!/usr/bin/env python3
""" RNNs """
import numpy as np


class RNNCell():
    """represents a cell of a simple RNN"""

    def __init__(self, i, h, o):
        """ constructor that initializes the weights and biases of the RNN
            of the i that is data h is the hidden state and o is the output
        """
        # tuplas
        self.Wh = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        performs forward propagation for one time step

        Args:
            h_prev: containing the previous hidden state,
            x_t: contains the data input for the cell

        Returns:
            h_next: containing the next hidden state
            y: contains the data output for the cell,
        """
        h_next = np.matmul(
            np.concatenate((h_prev, x_t), axis=1),
            self.Wh
        ) + self.bh
        y = np.matmul(h_next, self.Wy) + self.by

        return \
            np.tanh(h_next), \
            np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)
