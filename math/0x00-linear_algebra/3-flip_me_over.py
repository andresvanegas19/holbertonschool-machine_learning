#!/usr/bin/env python3
""" This is a script that contain a function
That will return a transpose of a 2D matrix """


def matrix_transpose(matrix):
    """ This will calculate and will returned a 2D array """
    transpose = []
    for i in range(len(matrix[0])):
        vector = []
        for row in matrix:
            vector += [row[i]]
        transpose += [vector]
    return transpose
