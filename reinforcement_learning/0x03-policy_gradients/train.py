#!/usr/bin/env python3
"""Program that implements a full training"""

import numpy as np
from policy_gradient import policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """
    implements a full training.

    Args:
        env: initial environment
        nb_episodes: number of episodes used for training
        alpha: the learning rate
        gamma: the discount factor

    Return:
        all values of the score (sum of all rewards during one episode loop)
    """
    weight = np.random.rand(4, 2)
    sco_reward = []

    for esc in range(nb_episodes):
        state = env.reset()[None, :]
        grads = []
        rewards = []
        v_score = 0

        while True:
            if show_result and (esc % 1000 == 0):
                env.render()

            action, grad = policy_gradient(state, weight)
            new_state, reward, done, _ = env.step(action)
            grads.append(grad)
            rewards.append(reward)
            v_score += reward
            state = new_state[None, :]

            if done:
                break

        for i in range(len(grads)):
            enum_s = enumerate(rewards[i:])
            weight += (
                alpha * grads[i] *
                sum([res * gamma**res for _, res in enum_s])
            )
        sco_reward.append(v_score)
        print("{}: {}".format(esc, v_score), end="\r", flush=False)

    return sco_reward
