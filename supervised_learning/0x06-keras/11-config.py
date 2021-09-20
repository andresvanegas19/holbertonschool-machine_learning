#!/usr/bin/env python3
""" Save and configuration of the model """


import tensorflow.keras as Keras


def save_config(network, filename):
    """save configuration of the model

    Args:
        network ([type]): [description]
        filename ([type]): [description]
    """
    with open(filename, 'w+') as file:
        file.write(network.to_json())


def load_config(filename):
    """Load configuration

    Args:
        filename ([type]): [description]

    Returns:
        [type]: [description]
    """
    with open(filename, 'r') as file:
        return Keras.models.model_from_json(file.read())