#!/usr/bin/env python3
""" Determinant module of a matrix in linear algebra """


def determinant(matrix):
    """
    calculates the determinant of a matrix

    Args:
        matrix (list of list):  whose determinant should be calculated

    Raises:
        TypeError: If matrix is not square
        ValueError: If matrix is not a list of lists

    Returns:
        [int]: the determinant of matrix
    """

    if type(matrix) is not list:
        raise TypeError("matrix must be a list of lists")

    if matrix == [[]]:
        return 1

    for i in range(len(matrix)):
        # check the b option in [[][]]
        if len(matrix) != len(matrix[i]):
            raise ValueError("matrix must be a square matrix")

        if type(matrix[i]) is not list or not len(matrix[i]):
            raise TypeError("matrix must be a list of lists")

    if len(matrix) == 1:
        return matrix[0][0]

    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    d = 0
    co = 1

    for i in range(len(matrix[0])):
        res = [vec[:] for vec in matrix]
        del res[0]

        for m in res:
            # r de vecto in de matrix
            del m[i]

        d += matrix[0][i] * determinant(res) * co
        co *= -1

    return d


def minor(matrix):
    """
    calculates the minor matrix of a matrix

    Args:
        matrix (list of list):  whose determinant should be calculated

    Raises:
        TypeError: If matrix is not square
        ValueError: If matrix is not a list of lists

    Returns:
        [int]: the determinant of matrix
    """

    if type(matrix) is not list or not len(matrix):
        raise TypeError("matrix must be a list of lists")

    if matrix == [[]]:
        raise ValueError("matrix must be a square matrix")

    for i in range(len(matrix)):
        if len(matrix) != len(matrix[i]):
            raise ValueError("matrix must be a square matrix")
        if type(matrix[i]) is not list or not len(matrix[i]):
            raise TypeError("matrix must be a list of lists")

    if len(matrix) == 1:
        return [[1]]

    minor = []

    for i in range(len(matrix)):
        pending = []
        for j in range(len(matrix)):
            mat = [vec[:] for vec in matrix]
            del mat[i]
            for line in mat:
                del line[j]
            det = determinant(mat)
            pending.append(det)
        minor.append(pending)
    return minor


def rs_ontacor(matrix):
    """[summary]

    Args:
        matrix ([type]): [description]

    Raises:
        TypeError: [description]
        ValueError: [description]
        ValueError: [description]
        TypeError: [description]

    Returns:
        [type]: [description]
    """
    if type(matrix) is not list or not len(matrix):
        raise TypeError("matrix must be a list of lists")

    if matrix == [[]]:
        raise ValueError("matrix must be a square matrix")

    for i in range(len(matrix)):
        if len(matrix) != len(matrix[i]):
            raise ValueError("matrix must be a square matrix")
        if type(matrix[i]) is not list or not len(matrix[i]):
            raise TypeError("matrix must be a list of lists")

    if len(matrix) == 1:
        return [[1]]

    minor = []

    for i in range(len(matrix)):
        resolve = []

        if i % 2 != 0:
            rs_ontac = -1
        else:
            rs_ontac = 1

        for j in range(len(matrix[0])):
            moch = [vec[:] for vec in matrix]
            del moch[i]
            for m in moch:
                del m[j]
            det = determinant(moch) * rs_ontac
            resolve.append(det)
            rs_ontac = rs_ontac * (-1)
        minor.append(resolve)

    return minor
