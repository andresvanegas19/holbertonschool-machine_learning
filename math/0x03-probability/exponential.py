#!/usr/bin/env python3
""" A model that  has a exponential distribution"""


class Exponential:
    """ for the exponential """

    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """
        Init the class where:
        data =  is a list of the data to be used to estimate the distribution
        lambtha = is the expected number of occurences in a given time frame
        """
        self.lambtha = float(lambtha)

        if data is None:
            if lambtha <= 0 or not lambtha:
                raise ValueError('lambtha must be a positive value')
            self.data = lambtha
            return

        if not isinstance(data, list):
            raise TypeError('data must be a list')

        if len(data) < 2:
            raise ValueError('data must contain multiple values')

        self.lambtha = float(1 / (sum(data) / len(data)))

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given time period
        x is the time period
        Returns the PDF value for x
        """

        if x is None or x < 0:
            return 0

        # for negative value put lambda in negative
        return self.lambtha * self.e ** (-1 * self.lambtha * x)

    def cdf(self, x):
        """
        Calculates the value of the CDF for a given time period
        x is the time period
        """

        if x is None or x < 0:
            return 0

        return 1 - self.e ** (-1 * self.lambtha * x)
