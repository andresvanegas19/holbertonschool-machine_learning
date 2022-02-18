#!/usr/bin/env python3
""" Temporal Difference """

import numpy as np


def monte_carlo(
    env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99
):
    """
        performs the Monte Carlo algorithm:

        Args:
            env: is the openAI environment instance
            V(numpy.ndarray):  containing the value estimate
            policy(function): takes in a state and returns the next action
            episodes: is the total number of episodes to train over
            max_steps(number): is the maximum number of steps per episode
            alpha: is the learning rate
            gamma: is the discount rate

        Returns:
            V, the updated value estimate
    """
    for i in range(episodes):
        env_res_s = env.reset()
        # print(env_res_s)
        episode = []

        for _ in range(max_steps):
            act_polic = policy(env_res_s)
            # print(act_polic)
            env_res_s_new, reward, done, _ = env.step(act_polic)
            episode.append(
                # another way to do this is to use env.render() []
                [env_res_s, act_polic, reward, env_res_s_new]
            )
            if done:
                break

            env_res_s = env_res_s_new

        G = 0
        # print(episode)
        # print(episode[-1])
        # sf = env.reset()
        episode = np.array(episode, dtype=int)

        for _, step in enumerate(episode[::-1]):
            # ef = episode[::-1]
            env_res_s, act_polic, reward, _ = step
            # STEP 1: Update the value function
            # G = reward + gamma * V[env_res_s] value function
            G = (gamma * G) + reward
            # print(G)
            if env_res_s not in episode[:i, 0]:
                # print(env_res_s)
                V[env_res_s] = V[env_res_s] + alpha * (G - V[env_res_s])

    return V
