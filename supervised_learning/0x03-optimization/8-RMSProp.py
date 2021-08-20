#!/usr/bin/env python3
""" This module contains the optimization methods """

import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    creates the training operation for a neural network in
    tensorflow using the RMSProp optimization algorithm:

    loss is the loss of the network
    alpha is the learning rate
    beta2 is the RMSProp weight
    epsilon is a small number to avoid division by zero

    Returns: the RMSProp optimization operation

    """

    rms = tf.train.RMSPropOptimizer(alpha, beta2, epsilon=epsilon)

    # compute gradient and apply
    return rms.apply_gradients(rms.compute_gradients(loss))
