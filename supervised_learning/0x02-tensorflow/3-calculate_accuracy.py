#!/usr/bin/env python3
""" Module for neural network """
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """
    calculates the accuracy of a prediction:

    y is a placeholder for the labels of the input data
    y_pred is a tensor containing the network’s predictions

    Returns: a tensor containing the decimal accuracy of the prediction
    """

    prediction = tf.argmax(y_pred, 1)

    equality = tf.equal(
        tf.argmax(y, 1),
        prediction
    )

    return tf.reduce_mean(tf.cast(equality, tf.float32))
