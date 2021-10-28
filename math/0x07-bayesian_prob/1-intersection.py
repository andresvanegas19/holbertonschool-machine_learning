#!/usr/bin/env python3
""" Do Bayesian Probability """


import numpy as np


def intersection(x, n, P, Pr):
    """
    calculates the marginal probability of obtaining the data
    in the Bayesian network P given the evidence x

    Args:
        x is the number of patients that develop severe side effects
        n is the total number of patients observed
        P is a 1D numpy.ndarray containing the various hypothetical
        probabilities of patients developing severe side effects
        Pr is a 1D numpy.ndarray containing the prior beliefs about P

    Returns:
        the marginal probability of obtaining x and n is the product of
        the prior beliefs Pr and the likelihood of x given n
    """
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")

    if type(x) is not int or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
        )

    if x > n:
        raise ValueError("x cannot be greater than n")

    if type(P) is not np.ndarray or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if type(Pr) is not np.ndarray or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    for pTe in P:
        if pTe < 0 or pTe > 1:
            raise ValueError(
                "All values in P must be in the range [0, 1]"
            )

    for prior in Pr:
        if prior < 0 or prior > 1:
            raise ValueError(
                "All values in Pr must be in the range [0, 1]"
            )

    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    # Compute the likelihood of x given n
    likelihood = \
        P ** x * (np.math.factorial(n) / (np.math.factorial(x) * np.math.factorial(n-x))) \
        * (1-P) ** (n-x)

    return likelihood * Pr
