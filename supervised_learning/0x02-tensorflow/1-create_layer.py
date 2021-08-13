#!/usr/bin/env python3
""" Module for neural network """
import tensorflow as tf


def create_layer(prev, n, activation):
    """
    Create a layer

    prev is the tensor output of the previous layer
    n is the number of nodes in the layer to create
    activation is the activation function that the layer should use
    use mode="FAN_AVG" to implement He et. al
    initialization for the layer weights
    each layer should be given the name layer

    Returns: the tensor output of the layer
    """
    # Initial the kernel
    k = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")

    # make a linear model
    return tf.layers.Dense(
        n,
        activation=activation,
        kernel_initializer=k,
        name="layer"
    )(prev)
