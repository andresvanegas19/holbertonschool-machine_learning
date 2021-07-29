#!/usr/bin/env python3
""" In this model Initialize Class Binomial """


class Binomial:
    """ represents a binomial distribution """

    e = 2.7182818285
    pi = 3.1415926536

    def __init__(self, data=None, n=1, p=0.5):
        """
        represents a binomial distribution

        data is a list of the data to be used to estimate the distribution
        n is the number of Bernoulli trials
        p is the probability of a “success”
        Sets the instance attributes n and p
        Saves n as an integer and p as a float
        """
        self.n = n
        self.p = p

        if data is not None and not isinstance(data, list):
            raise TypeError("data must be a list")

        if isinstance(data, list) and len(data) < 2:
            raise ValueError("data must contain multiple values")

        if p <= 0 or p >= 1:
            raise ValueError("p must be greater than 0 and less than 1")

        if n < 1:
            raise ValueError("n must be a positive value")

        if data is not None:
            ind = sum(data) / len(data)

            total = 0

            for x in data:
                total = total + (x - ind) ** 2

            self.n = int(
                ind / (1 - ((total / len(data)) / ind))
            )

            self.p = ind / self.n

    def pmf(self, k):
        """ probability method for binomial """
        if not isinstance(k, int):
            k = int(k)

        if k < 0:
            return 0

        n_fact = 1
        for i in range(1, self.n + 1):
            n_fact *= i

        x = 1
        for i in range(1, k + 1):
            x *= i

        fc_cp = 1
        for i in range(1, (self.n - k) + 1):
            fc_cp *= i

        xy_fact = (n_fact) / (x * fc_cp)

        return xy_fact * (self.p ** k) * ((1 - self.p) ** (self.n - k))

    def cdf(self, k):
        """ cumulative distribution for binomial """
        if not isinstance(k, int):
            k = int(k)

        if k < 0 or k > self.n:
            return 0

        cdf = 0
        for i in range(0, k + 1):
            cdf += self.pmf(i)

        return cdf
