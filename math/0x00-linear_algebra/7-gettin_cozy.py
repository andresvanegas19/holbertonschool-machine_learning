#!/usr/bin/env python3
""" This script will concatenate two arrays depends on the axis"""


def check_axis_0(mat1, mat2):
    """
    This function will check the axis of the matrix
    """

    new_matrix = []

    for row in mat1:
        new_matrix.append(row[:])

    return new_matrix + mat2


def check_axis_1(mat1, mat2, len_mat):
    """
    This function will check the axis of the matrix
    """
    if len_mat != len(mat1):
        return None

    new_matrix = []

    for i in range(len_mat):
        new_matrix.append(mat1[i] + mat2[i])

    return new_matrix


def cat_matrices2D(mat1, mat2, axis=0):
    """ This function will concatenates two arrays and
    will concatenate depends on the axis the have"""

    if axis == 0:
        if len(mat2[0]) != len(mat1[0]):
            return None
        return check_axis_0(mat1, mat2)

    elif axis == 1:
        return check_axis_1(mat1, mat2, len(mat2))

    return None
