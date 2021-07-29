#!/usr/bin/env python3
""" Initialize Poisson """


class Poisson:
    """ Class to make a Poisson function
        stastics
    """
    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """ Init the class where:
            data =  is a list of the data to be used
            to estimate the distribution
            lambtha = is the expected number of
            occurences in a given time frame
        """

        if not lambtha or lambtha <= 0:
            raise ValueError("lambtha must be a positive value")
        self.lambtha = float(lambtha)

        if data is not None:
            if not isinstance(data, list):
                raise TypeError("data must be a list")

            if len(data) <= 1:
                raise ValueError("data must contain multiple values")

            self.lambtha = float(sum(data) / len(data))
            self.data = data
        else:
            # If data is not given, Use the given lambtha
            self.data = float(lambtha)

    def pmf(self, k):
        """ Function to get the PMF of Poisson and
            Calculates the value of the PMF for
            a given number of “successes”
        """
        if k <= 0:
            return 0

        k = int(k)
        factorial_k = 1

        for i in range(1, k + 1):
            factorial_k *= i

        return (self.lambtha ** k) * (self.e ** -self.lambtha) / factorial_k

    def cdf(self, k):
        """
        Calculates the value of the CDF for a given number of “successes”
        Returns the CDF value for k
        """
        if not isinstance(k, int):
            k = int(k)

        if k < 0:
            return 0

        result = 0.0

        while k >= 0:
            result += self.pmf(k)
            # rest with this cause is not precise
            k -= 0.9999999999

        return result
