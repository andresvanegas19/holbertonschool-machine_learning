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

    dg_v = np.diag(P)
    if (dg_v == 1).all():
        return True

    b_inf = (dg_v == 1)
    for x in range(len(dg_v)):
        for y in range(len(dg_v)):
            if P[x, y] > 0 and b_inf[y]:
                b_inf[x] = 1

    if (b_inf == 1).all():
        return True

    return False
