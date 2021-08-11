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

        self.__W1 = np.random.normal(loc=0.0, scale=1.0, size=(nodes, nx))
        self.__b1 = np.zeros(nodes).reshape(nodes, 1)
        self.__A1 = 0
        self.__W2 = np.random.normal(
            loc=0.0, scale=1.0, size=nodes
        ).reshape(1, nodes)

        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """
        W1 The weights vector for the hidden layer. Upon instantiation,
        it should be initialized using a random normal distribution.

        Return: W1
        """
        return self.__W1

    @property
    def b1(self):
        """
        b1: The bias for the hidden layer. Upon instantiation,
        it should be initialized with 0’s.

        Return: b1 private
        """
        return self.__b1

    @property
    def A1(self):
        """
        A1: The activated output for the hidden layer.
        Upon instantiation, it should be initialized to 0.

        Return: A1
        """

        return self.__A1

    @property
    def W2(self):
        """
        W2: The weights vector for the output neuron.
        Upon instantiation, it should be initialized using a
        random normal distribution.

        Return: W2
        """
        return self.__W2

    @property
    def b2(self):
        """
        b2: The bias for the output neuron. Upon instantiation,
        it should be initialized to 0.

        Return: b2
        """
        return self.__b2

    @property
    def A2(self):
        """
        A2: The activated output for the output neuron
        (prediction). Upon instantiation, it should be initialized to 0.

        Return: A2
        """
        return self.__A2
