#!/usr/bin/env python3
""" RNNs """
import numpy as np


class GRUCell():
    """ Represents a gated recurrent unit """

    def __init__(self, i, h, o):
        """ GRUCell class constructor """
        self.Wz = np.random.normal(size=(i + h, h))
        self.Wr = np.random.normal(size=(i + h, h))
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))

        self.bh = np.zeros((1, h))
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        performs forward propagation for one time step

        Args:
            x_t is a numpy.ndarray of shape (m, i) that contains the data input
                for the cell
                - m is the batch size for the data
            h_prev is a numpy.ndarray of shape (m, h) containing the previous
                   hidden state

        Returns: h_next, y
            - h_next is the next hidden state
            - y is the output of the cell
        """
        concat1 = np.concatenate((h_prev, x_t), axis=1)

        # z_t = sigmoid(Wz @ concat1 + bz)
        r = np.matmul(concat1, self.Wr) + self.br
        r = 1 / (1 + np.exp(-r))

        # h_tilde = tanh(Wh @ concat1 + bh)
        z = np.matmul(concat1, self.Wz) + self.bz
        z = 1 / (1 + np.exp(-z))

        # h_t = z_t * h_prev + (1 - z_t) * h_tilde
        concat2 = np.concatenate((r * h_prev, x_t), axis=1)
        h_imp = np.matmul(concat2, self.Wh) + self.bh

        # y = softmax(Wy @ h_t + by)
        # h_next = z_t * h_prev + (1 - z_t) * h_tilde
        h_next = (1 - z) * h_prev + z * np.tanh(h_imp)

        y = np.matmul(h_next, self.Wy) + self.by
        y = np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)

        return h_next, y
