#!/usr/bin/env python3
""" Module for train a deep learning model with keras """

import tensorflow.keras as Keras


def save_model(network, filename):
    """[summary]

    Args:
        network ([type]): [description]
        filename ([type]): [description]

    Returns:
        [type]: [description]
    """
    network.save(filename)
    return None


def load_model(filename):
    """[summary]

    Args:
        filename ([type]): [description]

    Returns:
        [type]: [description]
    """
    return Keras.models.load_model(filename)
