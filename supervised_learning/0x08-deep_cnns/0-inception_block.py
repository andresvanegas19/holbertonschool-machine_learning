#!/usr/bin/env python3
""" Deep Convolutional Architectures """

import tensorflow.keras as K


def inception_block(A_prev, filters):
    """ [TAKE IN ACOUNT]
    All convolutions inside the inception block should use
    a rectified linear activation (ReLU)

    builds an inception block as described in Going
    Deeper with Convolutions:

    A_prev is the output from the previous layer
    filters is list containing F1, F3R, F3,F5R, F5, FPP, respectively:
        - F1 is the number of filters in the 1x1 convolution
        - F3R is the number of filters in the 1x1
            convolution before the 3x3 convolution
        - F3 is the number of filters in the 3x3 convolution
        - F5R is the number of filters in the 1x1
            convolution before the 5x5 convolution
        - F5 is the number of filters in the 5x5 convolution
        - FPP is the number of filters in the
            1x1 convolution after the max pooling
    Returns: the concatenated output of the inception block
    """
    F1, F3R, F3, F5R, F5, FPP = filters

    conv = K.layers.Conv2D(
        F1,
        1,
        activation='relu',
        kernel_initializer='he_normal'
    )(A_prev)

    conv_1 = K.layers.Conv2D(
        F3R,
        1,
        activation='relu',
        kernel_initializer='he_normal'
    )(A_prev)

    convd_2 = K.layers.Conv2D(
        F3,
        3,
        padding='same',
        activation='relu',
        kernel_initializer='he_normal'
    )(conv_1)

    conv_3 = K.layers.Conv2D(
        F5R, 1,
        activation='relu',
        kernel_initializer='he_normal'
    )(A_prev)

    conv_4 = K.layers.Conv2D(
        F5,
        5,
        padding='same',
        activation='relu',
        kernel_initializer='he_normal'
    )(conv_3)

    max_pooling = K.layers.MaxPool2D(
        3,
        1,
        padding='same'
    )(A_prev)

    last_layer = K.layers.Conv2D(
        FPP,
        1,
        activation='relu',
        kernel_initializer='he_normal'
    )(max_pooling)

    return K.layers.concatenate(
        [conv, convd_2, conv_4, last_layer]
    )
