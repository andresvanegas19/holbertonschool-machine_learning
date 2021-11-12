#!/usr/bin/env python3
""" Hidden Markov Models """

import numpy as np


def absorbing(P):
    """
    determines the absorbing state in a hidden markov model

    args:
        P: is a square 2D numpy.ndarray of shape (n, n) representing

    return:
        retruns a bool that means if the matrix is absorbing or not
    """
    if not isinstance(P, np.ndarray) \
            or len(P.shape) != 2 \
            or P.shape[0] != P.shape[1] \
            or P.shape[0] < 1:
        return None

    if np.all(np.diag(P) == 1):
        return True

    if P[0, 0] != 1:
        return False

    if np.all(np.count_nonzero(P[1:, 1:], axis=0) > 2):
        return True

    return False
