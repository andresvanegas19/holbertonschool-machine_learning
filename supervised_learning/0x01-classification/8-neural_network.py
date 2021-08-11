#!/usr/bin/env python3
""" Script that declared a class of neural network """
import numpy as np


class NeuralNetwork:
    """ An class NeuralNetwork  """

    def __init__(self, nx, nodes):
        """
        Initialize the main var
        """

        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')

        if nx < 1:
            raise ValueError('nx must be a positive integer')

        if not isinstance(nodes, int):
            raise TypeError('nodes must be an integer')

        if nodes < 1:
            raise ValueError('nodes must be a positive integer')

        self.W1 = np.random.normal(loc=0.0, scale=1.0, size=(nodes, nx))
        self.b1 = np.zeros(nodes).reshape(nodes, 1)
        self.W2 = np.random.normal(
            loc=0.0, scale=1.0, size=nodes
        ).reshape(1, nodes)

        self.nodes = nodes
        self.nx = nx
        self.A1 = 0
        self.b2 = 0
        self.A2 = 0
