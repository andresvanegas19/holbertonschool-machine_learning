#!/usr/bin/env python3
""" Do Bayesian Probability """


import numpy as np


def likelihood(x, n, P):
    """
    evolutionary cancer drug and are looking
    to find the probability that a patient who takes this drug will
    develop severe side effects. During your trials, n patients take the
    drug and x patients develop severe side effects. You can assume that
    x follows a binomial distribution.

    Args:
        x is the number of patients that develop severe side effects
        n is the total number of patients observed
        P is a 1D numpy.ndarray containing the various hypothetical
        probabilities of developing severe side effects


    Returns:
        a 1D numpy.ndarray containing the likelihood of obtaining the data, x
        and n, for each probability in P, respectively
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

    # pte is the probability of x or less patients
    for pte in P:
        if pte < 0 or pte > 1:
            raise ValueError("All values in P must be in the range [0, 1]")

    res = np.math.factorial(n) / (
        np.math.factorial(x) * np.math.factorial(n - x)
    )

    return res * (P ** x) * (1 - P) ** (n - x)
