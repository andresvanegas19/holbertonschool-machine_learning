#!/usr/bin/env python3
""" Module for train a deep learning model with keras """
import tensorflow.keras as Keras


def predict(network, data, verbose=False):
    """
    makes a prediction using a neural network:

    network is the network model to make the prediction with
    data is the input data to make the prediction with
    verbose is a boolean that determines if output should be
    printed during the prediction process

    Returns: the prediction for the data
    """

    return network.predict(data, verbose=verbose)
