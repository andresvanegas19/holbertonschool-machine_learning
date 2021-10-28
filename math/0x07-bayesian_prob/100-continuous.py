#!/usr/bin/env python3
""" Do Bayesian Probability """


from scipy import special


def posterior(x, n, p1, p2):
    """
    is the posterior probability that the
    probability of developing severe

    Args:
        x ([type]): [description]
        n ([type]): [description]
        p1 ([type]): [description]
        p2 ([type]): [description]


    Returns:
        the posterior probability that the probability of developing severe
    """
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")

    if type(x) is not int or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
        )

    if x > n:
        raise ValueError("x cannot be greater than n")

    if type(p1) is not float or p1 < 0 or p1 > 1:
        raise ValueError("p1 must be a float in the range [0, 1]")

    if type(p2) is not float or p2 < 0 or p2 > 1:
        raise ValueError("p2 must be a float in the range [0, 1]")

    if p2 <= p1:
        raise ValueError("p2 must be greater than p1")

    alpha = x + 1
    beta = n - x + 1

    return special.btdtr(alpha, beta, p1) - special.btdtr(alpha, beta, p2)
