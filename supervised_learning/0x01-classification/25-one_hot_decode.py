#!/usr/bin/env python3
""" module that converts a numeric label vector into a one-hot matrix """
import numpy as np


def one_hot_decode(one_hot):
    """
    that converts a numeric label vector into a one-hot matrix

    Y is a numpy.ndarray with shape (m,) containing numeric class labels
        - m is the number of examples
    classes is the maximum number of classes found in Y

    Returns: a one-hot encoding of Y with shape (classes, m)
    """

    if not isinstance(one_hot, np.ndarray) or len(one_hot.shape) is not 2:
        return None

    try:
        return np.argmax(one_hot, axis=0)
    except Exception:
        # avoid error IndexError: index 5 is out of bounds
        # for axis 0 with size 1
        return None
