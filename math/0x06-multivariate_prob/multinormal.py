#!/usr/bin/env python3
""" multinormal """

import numpy as np


class MultiNormal:
    """ Class that calculates the PDF at a data point  """

    def __init__(self, data):
        """ class constructor """

        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data, axis=1).reshape(data.shape[0], 1)
        cov = data - self.mean
        self.cov = np.dot(cov, cov.T) / (data.shape[1] - 1)

    def pdf(self, x):
        """ Calculates the PDF of the gaussian distribution """

        if not isinstance(x, np.ndarray):
            raise TypeError('x must be a numpy.ndarray')

        dimension, _ = self.cov.shape

        if len(x.shape) != 2 or x.shape[1] != 1 or x.shape[0] != dimension:
            raise ValueError(
                'x must have the shape ({}, 1)'.format(dimension)
            )

        dirc = np.dot(
            np.dot((x - self.mean).T, np.linalg.inv(self.cov)), (x - self.mean)
        )
        pdf = (1 / (
            ((2 * np.pi) ** (dimension / 2)) *
            (np.sqrt(np.linalg.det(self.cov)))
        )) * np.exp((-1 / 2) * dirc)
        return pdf[0][0]
