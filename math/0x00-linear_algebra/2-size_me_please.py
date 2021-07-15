#!/usr/bin/env python3
""" This script contain a function to shape a matrix """


def matrix_shape(matrix):
    """This calculate and returns  shape of a matrix """
    result = [len(matrix)]
    while 1:
        if isinstance(matrix[0], type(1)):
            break
        result += [len(matrix[0])]
        matrix = matrix[0]
    return result
