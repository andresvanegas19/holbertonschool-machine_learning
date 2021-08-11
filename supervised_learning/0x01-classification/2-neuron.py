#!/usr/bin/env python3
""" Create a class that simulate a neuron """

import numpy as np


class Neuron:
    """
    Main class that contains the all configuration of a neuron
    """

    def __init__(self, nx):
        """ nx is the number of input features to the neuron """

        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")

        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # A: The activated output of the neuron (prediction).
        # Upon instantiation, it should be initialized to 0.
        self.__A = 0
        # b: The bias for the neuron. Upon instantiation,
        # it should be initialized to 0.
        self.__b = 0
        # The weights vector for the neuron. Upon instantiation,
        # it should be initialized using a random normal distribution.
        self.__W = np.random.randn(1, nx)

    @property
    def A(self):
        """ getter method for the variable A"""
        return self.__A

    @property
    def W(self):
        """ getter method for the variable W"""
        return self.__W

    @property
    def b(self):
        """ getter method for the variable B"""
        return self.__b

    def forward_prop(self, X):
        """
        defines a single neuron performing binary classification and
        Calculates the forward propagation of the neuron

        The neuron should use a sigmoid activation function

        x = numpy array with shape (nx, m) that contains the input data
        nx is the number of input features to the neuron
        m is the number of examples


        Returns the private attribute __A
        """
        preactivation = np.matmul(self.__W, X) + self.__b

        #  forward propagation
        self.__A = 1 / (1 + np.exp(-preactivation))

        return self.__A
