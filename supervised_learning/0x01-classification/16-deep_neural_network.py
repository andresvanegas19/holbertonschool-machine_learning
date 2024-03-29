#!/usr/bin/env python3
""" Script that declared a class of deep neural network """
import numpy as np


class DeepNeuralNetwork:
    """ An class NeuralNetwork  """

    def __init__(self, nx, layers):
        """
        Initialize the main var
        """

        if not isinstance(layers, list):
            raise TypeError('layers must be a list of positive integers')

        if len(layers) == 0:
            raise TypeError('layers must be a list of positive integers')

        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')

        if nx < 1:
            raise ValueError('nx must be a positive integer')

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for i in range(self.L):

            if not isinstance(layers[i], int) or layers[i] <= 0:
                raise TypeError('layers must be a list of positive integers')

            if i == 0:
                self.weights['W' + str(i + 1)] = np.random.randn(
                    layers[i], nx
                ) * np.sqrt(2 / nx)

            else:
                self.weights['W' + str(i + 1)] = np.random.randn(
                    layers[i], layers[i - 1]
                ) * np.sqrt(2 / layers[i - 1])

            self.weights['b' + str(i + 1)] = np.zeros(
                layers[i]
            ).reshape(layers[i], 1)
