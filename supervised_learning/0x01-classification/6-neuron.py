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
        # np.matmul
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

        Returns the neuron’s prediction and the cost of the network,
        respectively
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

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neuron
        X is a numpy.ndarray with shape (nx, m) that contains the input data
        nx is the number of input features to the neuron
        m is the number of examples
        Y is a numpy.ndarray with shape (1, m) that contains the correct labels
        for the input data
        A is a numpy.ndarray with shape (1, m) containing the activated output
        of the neuron for each example
        alpha is the learning rate
        Updates the private attributes __W and __b
        """

        m = Y.shape[1]
        # z = w1X1 + w2X2 + b
        d = A - Y

        # gradient
        self.__W -= np.matmul(d, X.T) / m * alpha
        self.__b -= (np.sum(d) / m) * alpha

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Add the public method Trains the neuron

        X is a numpy.ndarray with shape (nx, m) that contains the input data
        nx is the number of input features to the neuron
        m is the number of examples
        Y is a numpy.ndarray with shape (1, m) that contains the correct labels
        for the input data
        iterations is the number of iterations to train over
        if iterations is not an integer, raise a TypeError with the exception
        iterations must be an integer
        if iterations is not positive, raise a ValueError with the exception
        iterations must be a positive integer
        alpha is the learning rate
        if alpha is not a float, raise a TypeError with the exception alpha
        must be a float
        if alpha is not positive, raise a ValueError with the exception alpha
        must be positive
        All exceptions should be raised in the order listed above
        Updates the private attributes __W, __b, and __A
        You are allowed to use one loop

        Returns the evaluation of the training data after
        iterations of training have occurred
        """
        # valide the data
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")

        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")

        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")

        if alpha <= 0:
            raise ValueError("alpha must be positive")

        for _ in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.A, alpha)

        return self.evaluate(X, Y)
