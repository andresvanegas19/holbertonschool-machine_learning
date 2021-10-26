#!/usr/bin/env python3
""" c of multinormal """

import numpy as np


class MultiNormal:
    """ Class that calculates the PDF at a data point  """

    def __init__(self, data):
        if type(data) is not np.ndarray or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")

        # init the declaration of the class
        self.mean = np.mean(data, axis=1).reshape((data.shape[0], 1))
        self.cov = np.matmul(data - self.mean, data.T) / (data.shape[1] - 1)

    def pdf(self, x):
        """ Calculates the PDF of the gaussian distribution """
        if type(x) != np.ndarray:
            raise TypeError('x must be a numpy.ndarray')

        dimension = self.cov.shape[0]

        if len(x.shape) != 2:
            raise ValueError(
                'x must have the shape ({}, 1)'.format(dimension)
            )

        if x.shape[0] != dimension or x.shape[1] != 1:
            raise ValueError(
                'x must have the shape ({}, 1)'.format(dimension)
            )

        dem = np.sqrt(
            ((2 * np.pi) ** x.shape[0]) * np.linalg.det(self.cov)
        )

        expo = -0.5 * np.matmul(
            np.matmul(
                (x - self.mean).T, np.linalg.inv(self.cov)
            ), x - self.mean
        )

        return (1 / dem) * np.exp(expo[0][0])
