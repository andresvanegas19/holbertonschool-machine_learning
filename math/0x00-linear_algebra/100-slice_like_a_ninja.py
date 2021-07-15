#!/usr/bin/env python3
""" slices a matrix along specific axes """


def np_slice(matrix, axes={}):
    """ slices a matrix along specific axes """
    slicer = [slice(None) for _ in range(len(matrix.shape))]

    for i, num in axes.items():
        slicer[i] = slice(*num)

    return matrix[tuple(slicer)]
