#!/usr/bin/env python3
""" Module for train a deep learning model with keras """
import tensorflow.keras as Keras


def one_hot(labels, classes=None):
    """
    converts a label vector into a one-hot matrix

    The last dimension of the one-hot matrix must be the number of classes
    Returns: the one-hot matrix
    """

    return Keras.utils.to_categorical(labels, classes)
