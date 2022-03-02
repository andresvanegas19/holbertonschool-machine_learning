#!/usr/bin/env python3
""" pandas """

import pandas as pd
# import numpy as np


def from_file(filename, delimiter):
    """
    loads data from a file as a pd.DataFrame
    Args:
        filename is the file to load from
        delimiter is the column separator

    Returns:
        the loaded pd.DataFrame
    """
    return pd.read_csv(filename, delimiter=delimiter)
