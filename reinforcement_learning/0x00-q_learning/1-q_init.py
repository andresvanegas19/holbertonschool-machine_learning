#!/usr/bin/env python3
""" reinforcement learning Q-learning """

import numpy as np


def q_init(env):
    """
    initializes the Q-table

    Args:
        env: is the FrozenLakeEnv instance

    Returns:
        the Q-table as a numpy.ndarray of zeros
    """

    return np.zeros(
        [  # Q-table shape
            env.observation_space.n,
            env.action_space.n
        ]
    )
