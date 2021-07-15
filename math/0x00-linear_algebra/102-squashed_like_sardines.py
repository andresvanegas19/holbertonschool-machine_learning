#!/usr/bin/env python3
""" This script will concatenate two arrays depends on the axis"""


def check_axis_concat(mat1, mat2, axis: int):
    """[summary]

    Args:
        mat1 ([type]): [description]
        mat2 ([type]): [description]
        axis (int): [description]

    Returns:
        [type]: [description]
    """
    if axis == 0:
        return mat1 + mat2

    return [
        check_axis_concat(mat1[i], mat2[i], axis - 1) for i in range(len(mat1))
    ]


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


def shape(matrix) -> bool:
    """
    calculate the shape of the matrix

    Args:
        matrix_one ([type]): [description]

    Returns:
        Return: the shape of the matrix
        bool: [description]
    """
    if isinstance(matrix[0], list):
        return [len(matrix)] + shape(matrix[0])

    return [len(matrix)]


def cat_matrices(mat1, mat2, axis=0):
    """
    This function will concatenates two arrays and
    will concatenate depends on the axis the have

    Args:
        mat1 ([type]): [description]
        mat2 ([type]): [description]
        axis (int, optional): [description]. Defaults to 0.

    Returns:
        [type]: [description]
    """
    if shape(mat1)[axis + 1:] != shape(mat2)[axis + 1:]:
        return None

    # if type(mat1) is not type(mat2):
    #     return False

    # if type(mat1) is not list:
    #     return False

    # if axis == 0:
    #     if len(mat2[0]) != len(mat1[0]):
    #         return None
    #     return check_axis_0(mat1, mat2)

    # elif axis == 1:
    #     return check_axis_1(mat1, mat2, len(mat2))

    return check_axis_concat(mat1, mat2, axis)

    return None
