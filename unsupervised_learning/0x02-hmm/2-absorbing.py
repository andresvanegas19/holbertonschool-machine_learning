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
    if type(P) is not np.ndarray or len(P.shape) != 2 \
            or P.shape[0] != P.shape[1] \
            or np.min(P ** 2) < 0 or np.min(P ** 3) < 0:
        return False

    sb = np.where(np.diag(P) == 1)
    if len(sb[0]) == P.shape[0]:
        return True
    if len(sb[0]) == 0:
        return False

    B = np.delete(np.delete(np.copy(P), sb[0], 0), sb[0], 1)
    In = np.identity(B.shape[0])

    try:
        np.linalg.inv(In - B)
        return True

    except Exception:
        return False
