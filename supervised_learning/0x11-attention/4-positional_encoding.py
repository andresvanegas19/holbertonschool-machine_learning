#!/usr/bin/env python3
""" Attention """
import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    calculates the positional encoding for a transformer

    Args:
        max_seq_len(integer): maximum sequence length
        dm: model depth

    Returns:
        a numpy.ndarray of shape (max_seq_len, dm) containing the positional
        encoding vectors

    """
    pos_vec = np.zeros((max_seq_len, dm))

    for position in range(max_seq_len):
        for i in range(0, dm, 2):
            div = np.exp(i * -np.log(10000.0) / dm)

            pos_vec[position, i] = (np.sin(position * div))
            pos_vec[position, i + 1] = (np.cos(position * div))

    return pos_vec
