#!/usr/bin/env python3
""" module that converts a numeric label vector into a one-hot matrix """
import numpy as np


def one_hot_encode(Y, classes):
    """
    that converts a numeric label vector into a one-hot matrix

    Y is a numpy.ndarray with shape (m,) containing numeric class labels
        - m is the number of examples
    classes is the maximum number of classes found in Y

    Returns: a one-hot encoding of Y with shape (classes, m)
    """

    if not isinstance(classes, int) or classes < 1:
        return None

    if not isinstance(Y, np.ndarray):
        return None

    try:
        encode = np.squeeze(np.eye(classes)[Y.reshape(-1)])
        return encode.T
    except Exception:
        # avoid error IndexError: index 5 is out of bounds
        # for axis 0 with size 1
        return None
