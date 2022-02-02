#!/usr/bin/env python3
""" reinforcement learning Q-learning """

import numpy as np


def play(env, Q, max_steps=100):
    """
    has the trained agent play an episode

    Args:
        env: the FrozenLakeEnv instance
        Q: numpy.ndarray containing the Q-table
        max_steps: maximum number of steps in the episode

    Returns:
        the total rewards for the episode
    """
    state = env.reset()
    env.render()
    for _ in range(max_steps):
        # Choose action
        action = np.argmax(Q[state, :])
        state, reward, done, _ = env.step(action)
        env.render()
        #

        if done:
            break

    return reward
