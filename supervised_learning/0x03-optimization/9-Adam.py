#!/usr/bin/env python3
""" This module contains the optimization methods """
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    updates a variable in place using the Adam optimization algorithm:

    alpha is the learning rate
    beta1 is the weight used for the first moment
    beta2 is the weight used for the second moment
    epsilon is a small number to avoid division by zero
    var is a numpy.ndarray containing the variable to be updated
    grad is a numpy.ndarray containing the gradient of var
    v is the previous first moment of var
    s is the previous second moment of var
    t is the time step used for bias correction

    Returns: the updated variable, the new first moment,
    and the new second moment, respectively
    """
    s = beta2 * s + (1 - beta2) * (grad * grad)
    corrected_sec = s / (1 - beta2 ** t)
    v = beta1 * v + (1 - beta1) * grad
    corrected_var = v / (1 - beta1 ** t)

    var -= alpha * corrected_var / (np.sqrt(corrected_sec) + epsilon)

    return var, v, s
