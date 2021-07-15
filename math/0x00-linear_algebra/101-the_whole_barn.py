#!/usr/bin/env python3
""" This is a script that contain a function
That will add two matrices """


def validate_matrix(matrix_one) -> bool:
    """[summary]

    Args:
        matrix_one ([type]): [description]

    Returns:
        bool: [description]
    """
    #     if isinstance(type(matrix_one[0]), list):
    #         return False

    #     if len(matrix_one) != len(matrix_two):
    #         return False

    #     return True

    i = len(matrix_one)
    matrix_dimensions = [i]

    while type(matrix_one[0]) == list:
        matrix_one = matrix_one[0]
        i = len(matrix_one)
        matrix_dimensions.append(i)

    return matrix_dimensions


# TODO: Modificated the documentation
def add_matrices(mat1, mat2):
    """[summary]

    Args:
        mat1 ([type]): [description]
        mat2 ([type]): [description]

    Returns:
        [type]: [description]
    """
    new_m = []

    if validate_matrix(mat1) != validate_matrix(mat2):
        return None

    if isinstance(mat1[0], type(1)) and len(mat1) == len(mat2):
        for i in range(len(mat1)):
            new_m += [mat1[i] + mat2[i]]
        return new_m

    for i in zip(mat1, mat2):
        suma = []
        matrix1, matrix2 = i
        if len(matrix1) != len(matrix2):
            return None
        for i in range(len(matrix1)):
            suma += [matrix1[i] + matrix2[i]]
        new_m += [suma]

    return new_m
