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

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.__L):

            if not isinstance(layers[i], int) or layers[i] <= 0:
                raise TypeError('layers must be a list of positive integers')

            if i == 0:
                self.__weights['W' + str(i + 1)] = np.random.randn(
                    layers[i], nx
                ) * np.sqrt(2 / nx)

            else:
                self.__weights['W' + str(i + 1)] = np.random.randn(
                    layers[i], layers[i - 1]
                ) * np.sqrt(2 / layers[i - 1])

            self.__weights['b' + str(i + 1)] = np.zeros(
                layers[i]
            ).reshape(layers[i], 1)

    @property
    def L(self):
        """ get the L """
        return self.__L

    @property
    def cache(self):
        """ get the cache """
        return self.__cache

    @property
    def weights(self):
        """
        get the wights
        A dictionary to hold all weights and biased of the network.
        Upon instantiation:
        The weights of the network should be initialized using the He
        et al. method and saved in the __weights dictionary using the
        key W{l} where {l} is the hidden layer the weight belongs to
        The biases of the network should be initialized to 0‘s and saved
        in the __weights dictionary using the key b{l} where {l} is the
        hidden layer the bias belongs to
        """
        return self.__weights

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
        # X should be saved to the cache dictionary using the key A0
        self.__cache['A0'] = X

        for i in range(self.L):

            Z = np.matmul(
                self.__weights["W" + str(i + 1)],
                self.__cache["A" + str(i)]
            ) + self.__weights["b" + str(i + 1)]
            v = 1 / (1 + np.exp(-Z))
            self.__cache["A" + str(i + 1)] = v

        return self.__cache["A" + str(self.L)], self.cache

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

        logaR = np.multiply(np.log(A), Y)
        logaF = np.multiply(np.log(1.0000001 - A), (1 - Y))

        return -(1 / Y.shape[1]) * np.sum(logaR + logaF)
