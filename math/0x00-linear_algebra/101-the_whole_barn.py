#!/usr/bin/env python3
""" This is a script that contain a function
That will add two matrices """


def add_matrices(mat1, mat2):
    """ that adds two matrices """
    new_m = []
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
