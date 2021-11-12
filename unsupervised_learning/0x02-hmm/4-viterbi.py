#!/usr/bin/env python3
""" Hidden Markov Models """

import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
    this vitirbi function returns the most likely sequence of hidden states

    args:
        Observation: is a numpy.ndarray of shape (T,) that contains the index
        Emission: is a numpy.ndarray of shape (N, M) containing the emission
        Transition: is a numpy.ndarray of shape (N, N) containing the
        Initial: is a numpy.ndarray of shape (N, 1) containing the initial

    Returns:
        P, B, or None, None on failure
    """
    if type(Observation) is not np.ndarray or len(Observation.shape) != 1:
        return (None, None)
    if type(Emission) is not np.ndarray or len(Emission.shape) != 2:
        return (None, None)
    if type(Transition) is not np.ndarray or len(Transition.shape) != 2:
        return (None, None)
    if type(Initial) is not np.ndarray or len(Initial.shape) != 2:
        return (None, None)

    T = Observation.shape[0]
    N, _ = Emission.shape
    if Transition.shape[0] != Transition.shape[1] or Transition.shape[0] != N:
        return (None, None)

    if N != Initial.shape[0] or Initial.shape[1] != 1:
        return (None, None)

    D = np.zeros([N, T])
    path = np.zeros(T)
    phi = np.zeros([N, T])
    D[:, 0] = Initial.T * Emission[:, Observation[0]]

    for i in range(1, T):
        for j in range(N):
            D[j, i] = np.max(D[:, i - 1] * Transition[:, j]) \
                * Emission[j, Observation[i]]

            phi[j, i] = np.argmax(D[:, i-1] * Transition[:, j])

    path[T - 1] = np.argmax(D[:, T - 1])

    for i in range(T-2, -1, -1):
        path[i] = phi[int(path[i + 1]), i + 1]

    return [int(i) for i in path], np.max(D[:, T - 1:], axis=0)[0]
