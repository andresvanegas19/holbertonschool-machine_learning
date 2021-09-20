#!/usr/bin/env python3
""" Deep Convolutional Architectures """

import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
    builds a transition layer as described in
    Densely Connected Convolutional Networks:

    X is the output from the previous layer
    nb_filters is an integer representing the number of filters in X
    compression is the compression factor for the transition layer

    implement compression as used in DenseNet-C
    All weights should use he normal initialization
    All convolutions should be preceded by
    Batch Normalization and a rectified linear
    activation (ReLU), respectively

    Returns: The output of the transition layer and
    the number of filters within the output, respectively
    """
    activation = K.layers.Activation('relu')(
        K.layers.BatchNormalization()(X)
    )

    conv2d = K.layers.Conv2D(
        int(nb_filters * compression),
        1,
        padding='same',
        kernel_initializer='he_normal'
    )(activation)

    av_pool_lay = K.layers.AveragePooling2D(
        2,
        strides=2,
        padding='same'
    )(conv2d)

    return av_pool_lay, av_pool_lay.shape[-1]
