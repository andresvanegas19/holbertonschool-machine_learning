#!/usr/bin/env python3
""" Deep Convolutional Architectures """


import tensorflow.keras as K

inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    builds the inception network as described in
    Going Deeper with Convolutions (2014):

    You can assume the input data will have shape (224, 224, 3)

    Returns: the keras model
    """
    initdeep = K.initializers.he_normal()

    # THE SHAPE OF THE INPUT
    X = K.Input(shape=(224, 224, 3))

    output_layer_1 = K.layers.Conv2D(
        filters=64,
        kernel_size=7,
        padding='same',
        strides=2,
        kernel_initializer=initdeep,
        activation='relu'
    )(X)

    output_layer_2 = K.layers.MaxPool2D(
        pool_size=3,
        padding='same',
        strides=2
    )(output_layer_1)

    layer_output_3R = K.layers.Conv2D(
        filters=64,
        kernel_size=1,
        padding='same',
        strides=1,
        kernel_initializer=initdeep,
        activation='relu'
    )(output_layer_2)

    layer_3 = K.layers.Conv2D(
        filters=192,
        kernel_size=3,
        padding='same',
        strides=1,
        kernel_initializer=initdeep,
        activation='relu'
    )(layer_output_3R)

    layer_4 = K.layers.MaxPool2D(
        pool_size=3,
        padding='same',
        strides=2
    )(layer_3)

    output_layer_5 = inception_block(layer_4, [64, 96, 128, 16, 32, 32])
    output_layer_6 = inception_block(
        output_layer_5, [128, 128, 192, 32, 96, 64])

    output_layer_7 = K.layers.MaxPool2D(
        pool_size=3,
        padding='same',
        strides=2
    )(output_layer_6)

    output_layer_8 = inception_block(
        output_layer_7, [192, 96, 208, 16, 48, 64]
    )
    output_layer_9 = inception_block(
        output_layer_8, [160, 112, 224, 24, 64, 64]
    )
    output_layer_10 = inception_block(
        output_layer_9, [128, 128, 256, 24, 64, 64]
    )
    output_layer_11 = inception_block(
        output_layer_10, [112, 144, 288, 32, 64, 64]
    )
    output_layer_12 = inception_block(
        output_layer_11, [256, 160, 320, 32, 128, 128])

    layer_output_13 = K.layers.MaxPool2D(
        pool_size=3,
        padding='same',
        strides=2
    )(output_layer_12)

    layer_output_14 = inception_block(
        layer_output_13, [256, 160, 320, 32, 128, 128])
    layer_output_15 = inception_block(
        layer_output_14, [384, 192, 384, 48, 128, 128])

    output_layer_16 = K.layers.AvgPool2D(
        pool_size=7,
        padding='same',
        strides=None
    )(layer_output_15)

    output_layer_17 = K.layers.Dropout(0.4)(output_layer_16)

    output_layer_18 = K.layers.Dense(
        units=1000,
        # here pass 'softmax' activation to the model
        activation='softmax',
        kernel_initializer=initdeep,
        kernel_regularizer=K.regularizers.l2()
    )(output_layer_17)

    return K.models.Model(inputs=X, outputs=output_layer_18)
