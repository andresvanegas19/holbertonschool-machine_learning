#!/usr/bin/env python3
""" Temporal Difference """

import numpy as np


def td_lambtha(
    env, V, policy, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99
):
    """
    performs the TD(λ) algorithm

    Args:
        env: openAI environment instance
        V(numpy.ndarray): containing the value estimate
        policy: is a function that takes in a state and returns the next action
        lambtha: is the eligibility trace factor
        episodes: is the total number of episodes to train over
        max_steps: is the maximum number of steps per episode
        alpha: is the learning rate
        gamma: is the discount rate

        Returns:
            V, the updated value estimate
    """
    space_size = env.observation_space.n
    temp = np.zeros(space_size)

    for _ in range(episodes):
        p_state = env.reset()
        for _ in range(max_steps):
            temp *= lambtha * gamma
            temp[p_state] += 1.0
            new_s, reward, done, _ = env.step(policy(p_state))

            # conditional validate G by the new state
            reward = env.desc.reshape(space_size)[
                new_s
            ] == b'G' if 1.0 else -1.0

            delta_t = reward + gamma * V[new_s] - V[p_state]
            V[p_state] = V[p_state] + alpha * delta_t * temp[p_state]

            if done:
                break

            p_state = new_s

    return V
