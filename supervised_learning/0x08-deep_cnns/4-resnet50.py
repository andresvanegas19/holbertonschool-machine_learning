#!/usr/bin/env python3
""" Deep Convolutional Architectures """


import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    builds the ResNet-50 architecture as described in
    Deep Residual Learning for Image Recognition (2015):

    You can assume the input data will have shape (224, 224, 3)
    All convolutions inside and outside the blocks should be
    followed by batch normalization along the channels
    axis and a rectified linear activation (ReLU), respectively.
    All weights should use he normal initialization

    Returns: the keras model
    """

    kernel_initializer = K.initializers.he_normal(seed=None)
    input_k = K.layers.Input(shape=(224, 224, 3))

    # Change the ambigues and the shape define with x for no errors
    # OVerride layer
    X = K.layers.Conv2D(
        filters=64,
        kernel_size=(7, 7),
        padding='same',
        strides=(2, 2),
        kernel_initializer=kernel_initializer
    )(input_k)

    X = K.layers.BatchNormalization()(X)
    X = K.layers.Activation('relu')(X)

    X = K.layers.MaxPooling2D((3, 3), padding='same', strides=(2, 2))(X)

    # Convu layer  2 stage
    X = projection_block(X, filters=[64, 64, 256], s=1)
    X = identity_block(X, [64, 64, 256])
    X = identity_block(X, [64, 64, 256])

    # Convu layer  2 stage
    X = projection_block(X, filters=[128, 128, 512], s=2)
    X = identity_block(X, [128, 128, 512])
    X = identity_block(X, [128, 128, 512])
    X = identity_block(X, [128, 128, 512])

    # Convu layer  2 stage
    X = projection_block(X, filters=[256, 256, 1024], s=2)
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])

    # Convu layer  2 stage
    X = projection_block(X, filters=[512, 512, 2048], s=2)
    X = identity_block(X, [512, 512, 2048])
    X = identity_block(X, [512, 512, 2048])

    X = K.layers.Dense(
        units=1000,
        activation='softmax',
        kernel_initializer=kernel_initializer
    )(K.layers.AveragePooling2D(pool_size=(7, 7), padding='same')(X))

    return K.models.Model(inputs=input_k, outputs=X, name='ResNet50')
