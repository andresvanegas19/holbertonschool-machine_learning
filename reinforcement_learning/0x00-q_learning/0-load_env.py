#!/usr/bin/env python3
""" reinforcement learning Q-learning """

import gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    loads the pre-made FrozenLakeEnv evnironment from OpenAI’s gym

    Args:
        desc: either None or a list of lists containing a custom description
            of the map to load for the environment
        map_name: either None or a string containing the
            pre-made map to load
        is_slippery: boolean to determine if the ice is slippery

    Returns:
        the environment
    """

    return gym.make(  # frozen lake environment
        'FrozenLake-v1',
        desc=desc,
        map_name=map_name,
        is_slippery=is_slippery
    )
