#!/usr/bin/env python3
""" Hidden Markov Models """

import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """
    performs the backward algorithm for a hidden markov model

    args:
        Observation: np.ndarray of shape (T,) that contains the index of the
        Emission: containing the emission probability of a specific
        observation given a hidden state
        Transition: np.ndarray of shape (N, N) containing the transition
        Initialized probability of a particular hidden state

    return:
        P: np.ndarray of shape (N, T) containing the backward path

    """
    if type(Observation) is not np.ndarray:
        return None, None

    if type(Emission) is not np.ndarray:
        return None, None

    if type(Transition) is not np.ndarray or len(Transition.shape) != 2:
        return None, None

    if Transition.shape[0] != Transition.shape[1]:
        return None, None

    N = Transition.shape[0]
    if type(Initial) is not np.ndarray or Initial.shape[0] != N:
        return None, None

    T = Observation.shape[0]

    # obsgm is the probability of the observation sequence given the model
    obsgm = np.zeros((N, T))
    obsgm[:, T - 1] = np.ones((N))

    for i in range(T - 2, -1, -1):
        for j in range(N):
            obsgm[j, i] = np.sum(
                obsgm[:, i + 1] *
                Transition[j, :] *
                Emission[:, Observation[i + 1]]
            )

    P = np.sum(Initial[:, 0] * Emission[:, Observation[0]] * obsgm[:, 0])
    return P, obsgm
