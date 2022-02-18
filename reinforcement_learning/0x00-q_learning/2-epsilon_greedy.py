#!/usr/bin/env python3
""" reinforcement learning Q-learning """

import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Uses epsilon-greedy to determine the next action.

    Args:
        Q: Tensor containing the q-table state is the current
            state.
        epsilon: The epsilon to use for the calculation.

    Returns:
        next (int): The next action index.
    """
    limit = np.random.uniform(0, 1)

    if epsilon < limit:  # epsilon-greedy
        index = np.argmax(Q[state, :])
    else:
        index = np.random.randint(Q.shape[1])

    return index
