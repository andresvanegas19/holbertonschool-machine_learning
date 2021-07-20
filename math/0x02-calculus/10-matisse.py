#!/usr/bin/env python3
""" Module that calculates the derivative of a polynomial """


def poly_derivative(poly):
    """
    calculates the derivative of a polynomial

    Returns: a new list of coefficients representing the
    derivative of the polynomial
    """

    if not poly or not isinstance(poly, list):
        return None

    if len(poly) <= 0:
        return None

    if len(poly) == 1:
        return [0]

    return [poly[i] * i for i in range(1, len(poly))]
