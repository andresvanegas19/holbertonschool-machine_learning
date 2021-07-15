#!/usr/bin/env python3
""" This is a script that contain a function
That will add two matrices """


def add_matrices2D(mat1, mat2):
    """ adds two matrices element-wise """
    new_m = []
    for i in zip(mat1, mat2):
        suma = []
        matrix1, matrix2 = i
        if len(matrix1) != len(matrix2):
            return None
        for i in range(len(matrix1)):
            suma += [matrix1[i] + matrix2[i]]
        new_m += [suma]
    return new_m
