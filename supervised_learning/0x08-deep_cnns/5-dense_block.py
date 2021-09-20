#!/usr/bin/env python3
""" Deep Convolutional Architectures """

import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    # from this paper https://arxiv.org/pdf/1608.06993.pdf
    """
    builds a dense block as described in Densely
    Connected Convolutional Networks:

    X is the output from the previous layer
    nb_filters is an integer representing the number of filters in X
    growth_rate is the growth rate for the dense block
    layers is the number of layers in the dense block

    use the bottleneck layers used for DenseNet-B
    All weights should use he normal initialization
    All convolutions should be preceded by Batch Normalization
    and a rectified linear activation (ReLU), respectively

    Returns: The concatenated output of each layer within the Dense
    Block and the number of filters within the
    concatenated outputs, respectively
    """
    kernel_initializer = K.initializers.he_normal()

    for _ in range(layers):

        activation = K.layers.Activation("relu")(
            K.layers.BatchNormalization()(X)
        )

        l_conv2d = K.layers.Conv2D(
            (4 * growth_rate),
            (1, 1),
            padding="same",
            kernel_initializer=kernel_initializer
        )(activation)

        activation_2 = K.layers.Activation("relu")(
            K.layers.BatchNormalization()(l_conv2d)
        )

        l_conv2d3 = K.layers.Conv2D(
            growth_rate, (3, 3),
            padding="same",
            kernel_initializer=kernel_initializer
        )(activation_2)

        X = K.layers.concatenate([X, l_conv2d3])

    return X, X.shape[-1]
