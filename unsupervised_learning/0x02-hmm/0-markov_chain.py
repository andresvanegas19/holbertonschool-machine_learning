#!/usr/bin/env python3
""" Hidden Markov Models """

import numpy as np


def markov_chain(P, s, t=1):
    """
    this function calculates the probability of a markov chain being in a
    particular state after a specified number of iterations using the
    forward algorithm

    Args:
        P: is a square 2D numpy.ndarray of shape (n, n) representing
        s: is a numpy.ndarray of shape (n, 1) representing the probability
        t: is the number of iterations that the markov chain

    Returns:
        a numpy.ndarray of shape (n, 1) representing the probability
    """

    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return None

    n, valN = P.shape

    if n != valN:
        return None

    if not isinstance(s, np.ndarray):
        return None

    if s.ndim != 2 or s.shape[0] != 1 or s.shape[1] != n:
        return None

    if not isinstance(t, int) or t < 1:
        return None

    for i in np.sum(P, axis=1):
        if not np.isclose(i, 1):
            return None

    valtmp = s
    tz_v = np.zeros((1, n))

    for i in range(t):
        tz_v = np.matmul(valtmp, P)
        valtmp = tz_v

    return tz_v
