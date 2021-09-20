#!/usr/bin/env python3
""" Deep Convolutional Architectures """

import tensorflow.keras as K


def identity_block(A_prev, filters):
    # from this paper https://arxiv.org/pdf/1512.03385.pdf
    """
    builds an identity block as described in Deep Residual Learning for
    Image Recognition (2015):

    A_prev is the output from the previous layer
    filters is a tuple or list containing F11, F3, F12, respectively:
        - F11 is the number of filters in the first 1x1 convolution
        - F3 is the number of filters in the 3x3 convolution
        - F12 is the number of filters in the second 1x1 convolution
    All convolutions inside the block should be followed by batch
    normalization along the channels axis and a rectified
    linear activation (ReLU), respectively.
    All weights should use he normal initialization

    Returns: the activated output of the identity block
    """
    kernel_initializer = K.initializers.he_normal()

    F11, F3, F12 = filters

    l_F11 = K.layers.Conv2D(
        F11, kernel_size=1, padding="same",
        kernel_initializer=kernel_initializer
    )(A_prev)

    s_F11 = K.layers.Activation("relu")(
        K.layers.BatchNormalization()(l_F11)
    )

    l_F3 = K.layers.Conv2D(
        F3, kernel_size=3, padding="same",
        kernel_initializer=kernel_initializer
    )(s_F11)

    norm_a_F3 = K.layers.Activation("relu")(
        K.layers.BatchNormalization()(l_F3)
    )

    ly_n_F12 = K.layers.Conv2D(
        F12,
        kernel_size=1,
        padding="same",
        kernel_initializer=kernel_initializer
    )(norm_a_F3)

    # NORM F12
    X = K.layers.Add()([K.layers.BatchNormalization()(ly_n_F12), A_prev])

    return K.layers.Activation("relu")(X)
