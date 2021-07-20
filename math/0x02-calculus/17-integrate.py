#!/usr/bin/env python3
""" Module that calculates the integral of a polynomial """


def poly_integral(poly, C=0):
    """
    calculates the integral of a polynomial
    poly is a list of coefficients representing a polynomial
        - the index of the list represents the power of x that
        the coefficient belongs to
    C is an integer representing the integration constant

    Return: a new list of coefficients representing the
    integral of the polynomial
    """

    # If poly or C are not valid, return None
    if not isinstance(poly, list) or len(poly) == 0 or not isinstance(C, int):
        return None

    if poly == [0]:
        return [C]

    poly_list = [C]
    for i in range(len(poly)):
        # Make the calculation
        if (poly[i] % (i + 1)) == 0:
            new = int(poly[i] / (i + 1))
        else:
            new = poly[i] / (i + 1)

        poly_list.append(new)

    return poly_list
