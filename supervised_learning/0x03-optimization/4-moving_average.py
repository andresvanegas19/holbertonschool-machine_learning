#!/usr/bin/env python3
""" This module contains the optimization methods """


def moving_average(data, beta):
    """
    calculates the weighted moving average of a data set:

    data is the list of data to calculate the moving average of
    beta is the weight used for the moving average
    Your moving average calculation should use bias correction

    Returns: a list containing the moving averages of data
    """

    if beta > 1 or beta < 0:
        return None

    ma = 0
    moving = []

    for i in range(len(data)):
        # Moving average
        ma = beta * ma + (1 - beta) * data[i]

        # Correction of bias
        correction = 1 - beta ** (i + 1)

        moving.append(ma / correction)

    return moving
