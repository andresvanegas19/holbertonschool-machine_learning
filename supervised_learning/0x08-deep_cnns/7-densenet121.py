#!/usr/bin/env python3
""" Deep Convolutional Architectures """

import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    builds the DenseNet-121 architecture as described in
    Densely Connected Convolutional Networks:

    growth_rate is the growth rate
    compression is the compression factor

    You can assume the input data will have shape (224, 224, 3)
    All convolutions should be preceded by Batch Normalization
    and a rectified linear activation (ReLU), respectively
    All weights should use he normal initialization

    Returns: the keras mode
    """

    input_data = K.layers.Input(shape=(224, 224, 3))
    kernel = K.initializers.he_normal(seed=None)
    X = K.layers.BatchNormalization()(input_data)
    X = K.layers.Activation('relu')(X)
    X = K.layers.Conv2D(
        filters=64,
        kernel_size=7,
        padding='same',
        strides=2,
        kernel_initializer=kernel
    )(X)

    X = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        padding='same',
        strides=(2, 2)
    )(X)

    X, fnl = dense_block(X, 2 * growth_rate, growth_rate, 6)
    X, fnl = transition_layer(X, fnl, compression)
    X, fnl = dense_block(X, fnl, growth_rate, 12)
    X, fnl = transition_layer(X, fnl, compression)
    X, fnl = dense_block(X, fnl, growth_rate, 24)
    X, fnl = transition_layer(X, fnl, compression)
    X, fnl = dense_block(X, fnl, growth_rate, 16)

    X = K.layers.AvgPool2D(
        pool_size=(7, 7),
        padding='same'
    )(X)

    X = K.layers.Dense(
        units=1000,
        activation='softmax',
        kernel_initializer=kernel
    )(X)

    return K.Model(inputs=input_data, outputs=X)
