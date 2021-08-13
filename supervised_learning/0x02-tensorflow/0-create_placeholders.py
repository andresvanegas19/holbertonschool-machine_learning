#!/usr/bin/env python3
""" Module for neural network """
import tensorflow as tf


def create_placeholders(nx, classes):
    """
    That create a place holder

    nx: the number of feature columns in our data
    classes: the number of classes in our classifier

    Returns: placeholders named x and y, respectively
    """
    # x is the placeholder for the input data to the neural network
    x = tf.placeholder(float, shape=[None, nx], name='x')
    # y is the placeholder for the one-hot labels for the input data
    y = tf.placeholder(float, shape=[None, classes], name='y')

    return x, y
