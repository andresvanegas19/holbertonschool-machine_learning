#!/usr/bin/env python3
""" Do the Dimensionality Reduction """
import numpy as np


def HP(Di, beta):
    """
    HP is the function that calculates the entropy of a distribution

    return: the entropy of a distribution
    """
    # ecuation of P(j|i)

    den = np.sum(np.exp(-Di.copy() * beta))
    Pi = np.exp(-Di.copy() * beta) / den

    # 1 / (2σ)^2
    return -np.sum(Pi * np.log2(Pi)), Pi
