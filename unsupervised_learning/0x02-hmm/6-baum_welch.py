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


def forward(Observation, Emission, Transition, Initial):
    """
    this function calculates the forward probability

    args:
        Observation: is a numpy.ndarray of shape (T,) that contains the index
        Emission: is a numpy.ndarray of shape (N, M) containing the emission
        Transition: is a numpy.ndarray of shape (N, N) containing the
        Initial: is a numpy.ndarray of shape (N, 1) containing the initial

    return (T, N) containing the forward probabilities
    """
    try:
        n_obs = Observation.shape[0]
        hdd_sta = Transition.shape[0]
        F = np.zeros((hdd_sta, n_obs))

        F[:, 0] = Initial.T * Emission[:, Observation[0]]

        for x in range(1, n_obs):
            for n in range(hdd_sta):
                F[n, x] = np.sum(
                    Transition[:, n] * F[:, x - 1] *
                    Emission[n, Observation[x]]
                )

        return np.sum(F[:, -1]), F

    except Exception:
        None, None


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    This is a function that performs the Baum-Welch algorithm for a hidden

    Args:
        Observations: is a list of observations
        Transition: is a numpy.ndarray of shape (N, N) containing the
        Emission: is a numpy.ndarray of shape (N, M) containing the emission
        Initial: is a numpy.ndarray of shape (N, 1) containing the initial
        iterations (int, optional): [description]. Defaults to 1000.

    Returns:
        returns the converged Transition, Emission, and Initial matrices
    """
    T = Observations.shape[0]
    # M is the number of possible observations
    M, N = Emission.shape

    for _ in range(1, iterations):
        F = forward(Observations, Emission, Transition, Initial)
        B = backward(Observations, Emission, Transition, Initial)
        xi = np.zeros((M, M, T - 1))

        for i in range(T - 1):
            den = np.dot(
                np.dot(F[:, i].T, Transition) *
                Emission[:, Observations[i + 1]].T, B[:, i + 1]
            )

            for j in range(M):
                num = F[j, i] * Transition[j] *\
                    Emission[:, Observations[i + 1]].T * B[:, i + 1].T
                xi[j, :, i] = num / den

        gm = np.sum(xi, axis=1)
        Transition = np.sum(xi, 2) / np.sum(gm, axis=1).reshape((-1, 1))
        gm = np.hstack(
            (
                gm,
                np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))
            )
        )
        denom = np.sum(gm, axis=1)

        for i in range(N):
            Emission[:, i] = np.sum(gm[:, Observations == i], axis=1)

    return Transition, np.divide(Emission, denom.reshape((-1, 1)))
