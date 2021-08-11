#!/usr/bin/env python3
""" Create a class that simulate a neuron """

import numpy as np


class Neuron:
    """
    Main class that contains the all configuration of a neuron
    defines a single neuron performing binary classification
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
        self.__W = np.random.normal(loc=0.0, scale=1.0, size=nx).reshape(1, nx)

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
        # not multiplicate with numpy make native
        val = np.matmul(self.__W, X) + self.__b

        #  forward propagation
        # self.__A = 1 / (1 + np.exp(np.matmul(self.__W, X) + self.__b))
        self.__A = 1 / (1 + np.exp(-1 * val))

        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        compute the cost

        Y is a numpy.ndarray with shape (1, m) that contains the correct labels
        for the input data

        A is a numpy.ndarray with shape (1, m) containing the activated output
        of the neuron for each example

        Returns the cost
        """

        # To avoid division by zero errors, use 1.0000001 - A instead of 1 - A
        cost = - (Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))

        # revert the value
        return cost.mean()

    def evaluate(self, X, Y):
        """
        Evaluates the neuron’s predictions

        X is a numpy.ndarray with shape (nx, m) that contains the input data
            nx is the number of input features to the neuron
            m is the number of examples

        Y is a numpy.ndarray with shape (1, m)
        that contains the correct labels for the input data

        Returns the neuron’s prediction and the
        cost of the network, respectively
            - The prediction should be a numpy.ndarray with shape (1, m)
                containing the predicted labels for each example
            - The label values should be 1 if the output of the
                network is >= 0.5 and 0 otherwise
        """

        # Generate activation and predict
        # The prediction should be a numpy.ndarray with shape (1, m)
        # containing the predicted labels for each example
        labels = np.where(self.forward_prop(X) < 0.5, 0, 1)

        # The label values should be 1 if the output of the network
        # is >= 0.5 and 0 otherwise
        return labels, self.cost(Y, self.__A)
