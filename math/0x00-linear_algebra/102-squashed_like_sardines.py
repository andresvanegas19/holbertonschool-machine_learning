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


def shape(matrix):
    """
    calculate the shape of the matrix
    Return: the shape of the matrix
    """
    shape = []
    while isinstance(matrix[0], list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape


def validate(mat1, mat2):
    """
    validate the matrix
    Return: the true if all the matrix are correct
    """
    if shape(mat1) != shape(mat2):
        return None
    if type(mat1) is not type(mat2):
        return None

    if type(mat1) is not list:
        return None

    return True


def cat_matrices(mat1, mat2, axis=0):
    """ This function will concatenates two arrays and
    will concatenate depends on the axis the have"""
    if not validate(mat1, mat2):
        return None

    if axis == 0:
        if len(mat2[0]) != len(mat1[0]):
            return None
        return check_axis_0(mat1, mat2)

    elif axis == 1:
        return check_axis_1(mat1, mat2, len(mat2))

    return None
