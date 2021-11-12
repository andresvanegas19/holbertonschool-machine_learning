#!/usr/bin/env python3
""" Hidden Markov Models """

import numpy as np


def regular(P):
    """
    this function calculates the likelihood of a sequence

    Arg:
        P: is a transition matrix with shape (n, n) representing the

    return:
        returns the likelihood of a sequence given the model
    """

    if (
        type(P) is not np.ndarray or
        len(P.shape) != 2 or
        P.shape[0] != P.shape[1]
    ):
        return None

    n = P.shape[0]

    tr = 0
    for i in range(1, 11):
        np.linalg.matrix_power(P, i)

        if (np.greater(P, 0).all()):
            tr = 1
            break

    if (not tr):
        return None

    # is it a regular HMM?
    cas = np.ones((n + 1, n))
    cas[:-1, :] = (P.T - np.eye(n))
    ded = np.ones((n + 1, 1))
    ded[:-1, :] = np.zeros((n, 1))

    X, _, _, _ = np.linalg.lstsq(
        a=cas,
        b=ded,
        rcond=None,
    )

    return X.reshape((1, n))
