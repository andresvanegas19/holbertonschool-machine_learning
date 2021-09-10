#!/usr/bin/env python3
""" Convolutional neural network """

import tensorflow.keras as K


def lenet5(x):
    """
    builds a modified version of the LeNet-5 architecture using keras:

    X is a K.Input of shape (m, 28, 28, 1)
    containing the input images for the network
        - m is the number of images
    The model should consist of the following layers in order:
        - Convolutional layer with 6 kernels of shape 5x5 with same padding
        - Max pooling layer with kernels of shape 2x2 with 2x2 strides
        - Convolutional layer with 16 kernels of shape 5x5 with valid padding
        - Max pooling layer with kernels of shape 2x2 with 2x2 strides
        - Fully connected layer with 120 nodes
        - Fully connected layer with 84 nodes
        - Fully connected softmax output layer with 10 nodes
    All layers requiring initialization should initialize their
    kernels with the he_normal initialization method
    All hidden layers requiring activation
    should use the relu activation function

    Returns: a K.Model compiled to use Adam optimization (with default
    hyperparameters) and accuracy metrics
    """
    # initialization for the layers weights
    initializer = K.initializers.he_normal(seed=None)

    layer_tf = K.layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        padding='same',
        activation='relu',
        kernel_initializer=initializer,
    )(x)

    # Max pooling layer with kernels of shape 2x2 with 2x2 strides
    layer_tf = K.layers.MaxPool2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(layer_tf)

    layer_tf = K.layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding='valid',
        activation='relu',
        kernel_initializer=initializer,
    )(layer_tf)

    layer_tf = K.layers.MaxPool2D(
        pool_size=(2, 2),
        strides=(2, 2),
    )(layer_tf)

    # Flattening between conv and dense layers
    layer_tf = K.layers.Flatten()(layer_tf)

    layer_tf = K.layers.Dense(
        units=120,
        activation='relu',
        kernel_initializer=initializer,
    )(layer_tf)

    layer_tf = K.layers.Dense(
        units=84,
        activation='relu',
        kernel_initializer=initializer,
    )(layer_tf)

    # Fully connected softmax output layer with 10 nodes
    layer_tf = K.layers.Dense(
        units=10,
        activation='softmax',
        kernel_initializer=initializer,
    )(layer_tf)

    return K.Model(inputs=x, outputs=layer_tf).compile(
        optimizer=K.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
