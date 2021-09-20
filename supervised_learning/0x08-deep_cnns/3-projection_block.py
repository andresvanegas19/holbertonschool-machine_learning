#!/usr/bin/env python3
""" Deep Convolutional Architectures """

import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    # from this paper https://arxiv.org/pdf/1512.03385.pdf
    """
    builds a projection block as described in
    Deep Residual Learning for Image Recognition (2015):

    A_prev is the output from the previous layer
    filters is a tuple or list containing F11, F3, F12, respectively:
        - F11 is the number of filters in the first 1x1 convolution
        - F3 is the number of filters in the 3x3 convolution
        - F12 is the number of filters in the second 1x1
            convolution as well as the 1x1 convolution in
            the shortcut connection
        - s is the stride of the first convolution
            in both the main path and the shortcut connection

    All convolutions inside the block should be followed
    by batch normalization along the channels axis and a
    rectified linear activation (ReLU), respectively.
    All weights should use he normal initialization

    Returns: the activated output of the projection block
    """
    F11, F3, F12 = filters

    lay_co1 = K.layers.Conv2D(
        F11,
        (1, 1),
        strides=s,
        padding="same",
        kernel_initializer="he_normal"
    )(A_prev)

    # NORM ALL WEIGHT
    activa_norm = K.layers.Activation(
        "relu"
    )(K.layers.BatchNormalization()(lay_co1))

    conv2d = K.layers.Conv2D(
        F3,
        (3, 3),
        padding="same",
        kernel_initializer="he_normal"
    )(activa_norm)

    activation_2 = K.layers.Activation("relu")(
        K.layers.BatchNormalization()(conv2d)
    )

    conv2d_3 = K.layers.Conv2D(
        F12,
        (1, 1),
        padding="same",
        kernel_initializer="he_normal"
    )(activation_2)

    conv2d_4 = K.layers.Conv2D(
        F12,
        (1, 1),
        s,
        padding="same",
        kernel_initializer="he_normal"
    )(A_prev)

    add = K.layers.Add()([
        K.layers.BatchNormalization()(conv2d_3),
        K.layers.BatchNormalization()(conv2d_4)
    ])

    return K.layers.Activation("relu")(add)
