#!/usr/bin/env python3
""" This script will performa the multiplication of a matrix"""


def calc_mul(column, row):
    """Calculate the multiplication"""
    pass


def mat_mul(mat1, mat2):
    """ that performs matrix multiplication"""

    # If the two matrices cannot be multiplied, return None
    if len(mat1[0]) != len(mat2):
        return None

    mult_m = []
    # that mat1 and mat2 are 2D matrices containing ints/floats
    # all elements in the same dimension are of the same type/shape
    for row in range(len(mat1)):
        vector = []
        for column in range(len(mat2[0])):

            # iterate for each vector into the matrix and multiplicate
            vector.append(
                sum(
                    # Create an array for all value in the vector for sum
                    [
                        mat2[loc][column] * mat1[row][loc]
                        for loc in range(len(mat1[0]))
                    ]
                )
            )

        mult_m.append(vector)

    return mult_m
