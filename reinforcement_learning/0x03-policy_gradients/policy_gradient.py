#!/usr/bin/env python3
""" Policy Gradients """
import numpy as np


def policy(matrix, weight):
    """
    computes to policy with a weight of a matrix.

    Args:
        matrix (np.array): the matrix to compute the policy
        weight (np.array): the weight matrix

    Returns:
        the policy
    """
    expo = np.exp(matrix.dot(weight))

    return expo / expo.sum()


def policy_gradient(state, weight):
    """
    computes the Monte-Carlo policy gradient based on a state
    and a weight matrix.

    Args:
        state: matrix representing the current observation of the environment
        weight: matrix of random weight

    Return:
        the action and the gradient (in this order)
    """
    policy_res = policy(state, weight)
    action = np.random.choice(len(policy_res[0]), p=policy_res[0])

    # compute the gradient
    po_res = policy_res.reshape(-1, 1)
    # s = np.array([[1, 0, 0, 0]])
    # s =3 * s
    softmax = (np.diagflat(po_res) - np.dot(po_res, po_res.T))[action, :]
    # softmax = softmax[action, :]
    dio_log = (softmax / policy_res[0, action])[None, :]
    gradient = state.T.dot(dio_log)

    return action, gradient
