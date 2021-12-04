#!/usr/bin/env python3
""" RNNs """
import numpy as np


class LSTMCell():
    """ Represents LSTM unit """

    def __init__(self, i, h, o):
        """ LSTMCell class constructor """
        self.Wf = np.random.normal(size=(i + h, h))
        self.Wu = np.random.normal(size=(i + h, h))
        self.Wc = np.random.normal(size=(i + h, h))
        self.Wo = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))

        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """
        performs forward propagation for one time step

        Args:
            x_t: contains the data input for the cell
            h_prev: containing the previous hidden state
            c_prev: containing the previous cell state

        Returns:
            - h_next is the next hidden state
            - c_next is the next cell state
            - y is the output of the cell
        """
        main_co = np.concatenate((h_prev, x_t), axis=1)

        rf = np.matmul(main_co, self.Wf) + self.bf
        # sigmoid expression
        rf = (1 / (1 + np.exp(-rf)))

        ru = np.matmul(main_co, self.Wu) + self.bu
        ru = (1 / (1 + np.exp(-ru)))
        # tanh expression
        rc = np.matmul(main_co, self.Wc) + self.bc
        rc = np.tanh(rc)
        ro = np.matmul(main_co, self.Wo) + self.bo
        # sigmoid expressions
        ro = (1 / (1 + np.exp(-ro)))

        c_next = (ru * rc) + (rf * c_prev)
        h_next = ro * np.tanh(c_next)

        y = np.matmul(h_next, self.Wy) + self.by
        y = np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)

        return h_next, c_next, y
