#!/usr/bin/env python3
""" script that make a sum without for """


def summation_i_squared(n):
    """
    calculates the sigma of i of up 2
    n is the stopping condition

    Return: the integer value of the sum
    """
    # If n is not a valid number, return None
    if isinstance(n, (int)) and n > 0:
        # form
        return int(n * (n + 1) * (2 * n + 1) / 6)
    else:
        return None
