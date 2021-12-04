#!/usr/bin/env python3
""" RNNs """
import numpy as np


class BidirectionalCell():
    """ Represents bidirectional cell of an RNN """

    def __init__(self, i, h, o):
        """ Class constructor for Bidirectional Cell """
        self.Whf = np.random.normal(size=(i + h, h))
        self.Whb = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=((2 * h), o))
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        calculates the hidden state in the forward direction for one time step

        Args:
            x_t: contains the data input for the cell
                - m is the batch size for the data
            h_prev: containing the previous hidden state

        Returns:
            the next hidden state
        """
        return np.tanh(
            np.concatenate(
                (h_prev, x_t),
                axis=1
            ) @ self.Whf + self.bhf
        )

    def output(self, H):
        """
        calculates all outputs for the RNN

        Args:
            H: contains the concatenated hidden states from both
                directions, excluding their initialized states
                - t is the number of time steps
                - m is the batch size for the data
                - h is the dimensionality of the hidden states

        Returns:
            the outputs
        """
        T, m, _ = H.shape
        Y = np.zeros((T, m, self.by.shape[-1]))

        for steps in range(T):
            Y[steps] = np.matmul(H[steps], self.Wy) + self.by
            Y[steps] = np.exp(Y[steps]) / \
                        np.sum(
                            np.exp(Y[steps]),
                            axis=1,
                            keepdims=True
                        )

        return Y
