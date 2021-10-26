#!/usr/bin/env python3
""" Determinant module of a matrix in linear algebra """

import numpy as np


def definiteness(matrix):
    """
    calculates the definiteness of a matrix:

    Args:
        matrix:  is a numpy.ndarray of shape (n, n)
        whose definiteness should be calculated

    Raises:
        TypeError: is not a matrix numpy

    Returns: Positive definite, Positive semi-definite,
        Negative semi-definite, Negative definite, or
        Indefinite if the matrix is positive definite,
        positive semi-definite, negative semi-definite,
        negative definite of indefinite, respectively
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    if len(matrix.shape) != 2:
        return None

    row, col = matrix.shape
    if row != col:
        return None

    if not (matrix.T == matrix).all():
        return None

    v_eig = np.linalg.eigvals(matrix)

    pos = 0
    neg = 0
    semi = 0
    for i in v_eig:
        if i > 0:
            pos = 1
        if i < 0:
            neg = 1
        if i == 0:
            semi = 1

    if pos and not semi and not neg:
        return "Positive definite"

    elif pos and semi and not neg:
        return "Positive semi-definite"

    elif not pos and not semi and neg:
        return "Negative definite"

    elif not pos and semi and neg:
        return "Negative semi-definite"

    elif pos and not semi and neg:
        return "Indefinite"

    else:
        return None
