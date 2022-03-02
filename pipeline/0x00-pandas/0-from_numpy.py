#!/usr/bin/env python3
""" pandas """

import pandas as pd
# import numpy as np


def from_numpy(array):
    """
    creates a pd.DataFrame from a np.ndarray

    Args:
        array(np.ndarray): from which you should create the pd.DataFrame

    Returns:
        the newly created pd.DataFrame
    """
    alphabet = list("ABCDEFGHIJKLMNOPQRSTUVW")

    return pd.DataFrame(
        array,
        columns=alphabet[:array.shape[1]]
    )
