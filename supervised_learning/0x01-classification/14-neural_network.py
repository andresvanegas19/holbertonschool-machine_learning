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
        # see the rendimize
        val_1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1/(1 + np.exp(-1 * val_1))

        val_2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1/(1 + np.exp(-1 * val_2))

        return self.A1, self.A2

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

        # To avoid division by zero errors, please use 1.0000001 - A
        # Calculates the cost of the model using logistic regression
        loss = -(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        )

        return np.sum(loss[0]) / Y.shape[1]

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
        self.forward_prop(X)

        return np.where(self.A2 >= 0.5, 1, 0), self.cost(Y, self.A2)

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        neural network with one hidden layer performing binary classification
        Calculates one pass of gradient descent on the neural network

        X is a numpy.ndarray with shape (nx, m) that contains the input data
        nx is the number of input features to the neuron
        m is the number of examples
        Y is a numpy.ndarray with shape (1, m) that contains the correct
        labels for the input data
        A1 is the output of the hidden layer
        A2 is the predicted output
        alpha is the learning rate
        """

        dW2 = (1 / Y.shape[1]) * np.matmul(A2 - Y, A1.T)
        db2 = (1 / Y.shape[1]) * np.sum(A2 - Y, axis=1, keepdims=True)

        dev_Z = np.multiply(np.matmul(self.W2.T, A2 - Y), (A1 * (1 - A1)))

        self.__W2 -= alpha * dW2
        self.__b2 -= alpha * db2
        self.__W1 -= alpha * (1 / Y.shape[1]) * np.matmul(dev_Z, X.T)
        self.__b1 -= alpha * (1 / Y.shape[1]) * \
            np.sum(dev_Z, axis=1, keepdims=True)

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Add the public method Trains the neuron

        Trains the neuron by updating the private attributes __W, __b, and __A

        graph is a boolean that defines whether or
        not to graphinformation about the training
        once the training has completed.

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

        Returns: the evaluation of the training
        data after iterations of training have occurred
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
            A1, A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, A1, A2, alpha)
        return self.evaluate(X, Y)
