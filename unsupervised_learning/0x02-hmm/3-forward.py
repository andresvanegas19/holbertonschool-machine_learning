#!/usr/bin/env python3
""" Hidden Markov Models """

import numpy as np


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
