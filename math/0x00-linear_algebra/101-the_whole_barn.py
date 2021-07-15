#!/usr/bin/env python3
"""
    This is a script that contain a function that will add two matrices
"""


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

    shape_matrix_one = validate_matrix(mat1)
    shape_matrix_two = validate_matrix(mat2)
    if shape_matrix_one != shape_matrix_two:
        return None

    # Add arrays
    # TODO: Pass all to a function
    if isinstance(mat1[0], list) and isinstance(mat2[0], list):
        if len(mat1) != len(mat2):
            return None

        result = []
        for i in range(len(mat1)):
            result.append(mat1[i] + mat2[i])
        return result

    # 2d
    if len(shape_matrix_one) == 2 and len(shape_matrix_two) == 2:
        if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
            return None

        return [
            [
                mat1[i][j] + mat2[i][j] for j in range(len(mat2[0]))
            ] for i in range(len(mat2))
        ]

    # if isinstance(mat1[0], type(1)) and len(mat1) == len(mat2):
    #     for i in range(len(mat1)):
    #         new_m += [mat1[i] + mat2[i]]
    #     return new_m

    # for i in zip(mat1, mat2):
    #     suma = []
    #     matrix1, matrix2 = i
    #     if len(matrix1) != len(matrix2):
    #         return None
    #     for i in range(len(matrix1)):
    #         suma += [matrix1[i] + matrix2[i]]
    #     new_m += [suma]

    # return new_m
    result = []
    for vec in range(len(mat1)):
        result.append(
            add_matrices(mat1[vec], mat2[vec])
        )
    return result
