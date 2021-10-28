#!/usr/bin/env python3
""" Do the Dimensionality Reduction """

import numpy as np


def Q_affinities(Y):
    """
    function is the Q affinities that performs the dimensionality reduction
    and returns the new data matrix Y.
    """
    rowY = np.sum(np.square(Y), 1)

    div = 1 + np.add(np.add(-2 * np.dot(Y, Y.T), rowY).T, rowY)

    # num is the numerator of the Q affinities
    num = 1 / div
    num[range(Y.shape[0]), range(Y.shape[0])] = 0

    return num / np.sum(num), num
