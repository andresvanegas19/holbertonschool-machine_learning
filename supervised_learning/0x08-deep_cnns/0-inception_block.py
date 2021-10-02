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
    F1 = filters[0]
    F3R = filters[1]
    F3 = filters[2]
    F5R = filters[3]
    F5 = filters[4]
    FPP = filters[5]
    padding = "same"

    he_init = K.initializers.he_normal()

    CF1 = K.layers.Conv2D(
        filters=F1, kernel_size=(1, 1),
        padding=padding,
        activation='relu',
        kernel_initializer=he_init
    )(A_prev)
    CF3R = K.layers.Conv2D(
        filters=F3R,
        kernel_size=(1, 1),
        padding=padding,
        activation='relu',
        kernel_initializer=he_init
    )(A_prev)
    CF3 = K.layers.Conv2D(
        filters=F3,
        kernel_size=3,
        padding=padding,
        activation='relu',
        kernel_initializer=he_init
    )(CF3R)
    CF5R = K.layers.Conv2D(
        filters=F5R,
        kernel_size=(1, 1),
        padding=padding,
        activation='relu',
        kernel_initializer=he_init
    )(A_prev)
    CF5 = K.layers.Conv2D(
        filters=F5,
        kernel_size=(5, 5),
        padding=padding,
        activation='relu',
        kernel_initializer=he_init
    )(CF5R)
    M_pooling = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=1,
        padding=padding
    )(A_prev)
    C_FPP = K.layers.Conv2D(
        filters=FPP,
        kernel_size=1,
        padding=padding,
        activation='relu',
        kernel_initializer=he_init
    )(M_pooling)

    zab = [CF1, CF3, CF5, C_FPP]

    return K.layers.concatenate(zab)
