#!/usr/bin/env python3
""" This module contains the optimization methods """
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    creates a batch normalization layer for a neural network in tensorflow:

    prev is the activated output of the previous layer
    n is the number of nodes in the layer to be created
    activation is the activation function that should be used
    on the output of the layer

    Returns: a tensor of the activated output for the layer
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, kernel_initializer=init, name='layer')

    net = layer(prev)

    gamma = tf.Variable(tf.constant(1.0, shape=[n]), trainable=True)
    mean, var = tf.nn.moments(net, axes=[0])
    beta = tf.Variable(tf.constant(0.0, shape=[n]), trainable=True)

    Z = tf.nn.batch_normalization(
        net,
        mean=mean,
        variance=var,
        offset=beta,
        scale=gamma,
        variance_epsilon=1e-8
    )
    return activation(Z)
