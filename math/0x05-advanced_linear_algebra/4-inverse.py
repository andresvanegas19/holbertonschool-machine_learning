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

    if len(matrix) == 1:
        return matrix[0][0]

    if len(matrix) == 2:
        return (
            (matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0])
        )

    result = 0
    for i, j in enumerate(matrix[0]):
        # change for no validate
        row = [v for v in matrix[1:]]
        appabs = []

        for num in row:
            a_num = []

            for l_max in range(len(matrix)):
                if l_max != i:
                    a_num.append(num[l_max])

            appabs.append(a_num)
        result = result + j * (-1) ** i * determinant(appabs)

    return result


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

    if not isinstance(matrix, list) or matrix == []:
        raise TypeError('matrix must be a list of lists')

    if any(not isinstance(row, list) for row in matrix):
        raise TypeError('matrix must be a list of lists')

    if any(len(row) != len(matrix) for row in matrix):
        raise ValueError('matrix must be a non-empty square matrix')

    if len(matrix) == 1:
        return [[1]]

    minor = []
    for i in range(len(matrix)):
        pending = []
        # move the a
        for j in range(len(matrix)):
            mat = [vec[:] for vec in matrix]
            del mat[i]
            # del { to v }
            for line in mat:
                del line[j]

            det = determinant(mat)
            pending.append(det)

        # append the minor
        minor.append(pending)

    return minor


def cofactor(matrix):
    """
    calculates the cofactor matrix of a matrix

    Args:
        matrix ([type]): [description]

    Returns:
        [type]: [description]
    """
    minx = minor(matrix)
    cf = minx.copy()

    for i in range(len(minx)):
        for j in range(len(minx)):
            cf[i][j] = cf[i][j] * (-1)**(i + j)
    return cf


def adjugate(matrix):
    """
    calculates the adjugate

    Args:
        matrix: a list of lists whose adjugate matrix should be calculated

    Returns: the adjugate of a matrix
    """
    adjgt = cofactor(matrix)

    res = []
    for j in range(len(adjgt[0])):
        u_n = []
        for i in range(len(adjgt)):
            u_n.append(adjgt[i][j])

        res.append(u_n)

    return res


def inverse(matrix):
    """
    calculates the inverse of a matrix

    Args:
        matrix: a list of lists whose adjugate matrix should be calculated

    Returns: the inversed matrix
    """

    # if type(matrix) != list or matrix == [] or type(matrix[0]) != list:
    #     raise TypeError('matrix must be a list of lists')

    # if matrix == [[]]:
    #     raise ValueError('matrix must be a non-empty square matrix')

    # for i in matrix:
    #     if type(i) != list:
    #         raise TypeError('matrix must be a list of lists')
    #     if len(i) != len(matrix):
    #         raise ValueError('matrix must be a non-empty square matrix')

    r_determinat = determinant(matrix)

    if r_determinat == 0:
        return None

    res = adjugate(matrix)

    for i in range(len(res)):
        for j in range(len(res)):
            res[i][j] = res[i][j] / r_determinat

    return res
