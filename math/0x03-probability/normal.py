#!/usr/bin/env python3
""" Initialize Poisson """


class Normal:
    """ class of the Normla """

    e = 2.7182818285
    pi = 3.1415926536

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        this is a test
        data is a list of the data to be used to estimate the distribution
        mean is the mean of the distribution
        stddev is the standard deviation of the distribution
        """

        if data is None:
            if stddev <= 0 or not stddev:
                raise ValueError('stddev must be a positive value')
            else:
                # Set to the basic parameters
                self.mean = float(mean)
                self.stddev = float(stddev)
                return

        if not isinstance(data, list):
            raise TypeError("data must be a list")

        if len(data) < 2:
            raise ValueError('data must contain multiple values')

        self.mean = sum(data) / len(data)
        self.stddev = (
            sum(map(lambda x: (x - self.mean) ** 2, data)) / len(data)
        ) ** 0.5

    def z_score(self, x):
        """ function to get the score """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """ function to get the score """
        return (z * self.stddev) + self.mean

    def pdf(self, x):
        """ get the x-value and calculate it of pdf"""
        coefficient = 1 / (self.stddev * (2 * self.pi) ** 0.5)
        expo = -0.5 * ((x - self.mean) / self.stddev) ** 2

        return coefficient * self.e ** (expo)

    def cdf(self, x):
        """ Make the functions """
        val = (x - self.mean) / (self.stddev * 2 ** .5)

        return (1 + (
            val - val ** 3 / 3 + val ** 5 / 10 - val ** 7 / 42 + val ** 9 / 216
        ) * 2 / (self.pi ** .5)
        ) / 2
