#!/usr/bin/env python3
""" Temporal Difference """
import numpy as np


def sarsa_lambtha(
        env, Q, lambtha, episodes=5000,
        max_steps=100, alpha=0.1, gamma=0.99,
        epsilon=1, min_epsilon=0.1, epsilon_decay=0.05
):
    """
        performs the TD(λ) algorithm

        env: is the openAI environment instance
        Q(numpy.ndarray): containing the Q table
        lambtha: is the elgibility trace factor
        episodes: is the total number of episodes to train over
        max_steps: is the maximum number of steps per episode
        alpha: is the learning rate
        gamma: is the discount rate
        epsilon: is the initial threshold for epsilon greedy
        min_epsilon: is the min value epsilon should decay to
        epsilon_decay: is decay rate for updating epsilon between episodes

        Returns:
            Q, the updated value estimate
    """
    action = 0
    int_eps = epsilon
    dec_eps = np.zeros(Q.shape)

    for i in range(episodes):
        ins_s = env.reset()

        action = np.random.uniform(0, 1) < int_eps \
            if env.action_space.sample() else np.argmax(Q[ins_s, :])

        for _ in range(max_steps):
            s_new, reward, complete, _ = env.step(action)

            # update Q
            action_new = np.random.uniform(0, 1) < int_eps \
                if env.action_space.sample() else np.argmax(Q[ins_s, :])
            # action_new = np.argmax(Q[s_new, :])
            dec_eps *= gamma * epsilon * lambtha
            dec_eps[ins_s, action] += (1.0)
            delta = reward + gamma * Q[s_new, action_new] - Q[ins_s, action]
            Q += alpha * delta * dec_eps

            if complete:
                break
            else:
                ins_s = s_new
                action = action_new

        if epsilon < min_epsilon:
            epsilon = min_epsilon
        else:
            epsilon *= int_eps * np.exp(-epsilon_decay * i)

    return Q
