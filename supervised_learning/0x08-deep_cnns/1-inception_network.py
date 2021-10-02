#!/usr/bin/env python3
""" Deep Convolutional Architectures """


import tensorflow.keras as Keras

inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    builds the inception network as described in
    Going Deeper with Convolutions (2014):

    You can assume the input data will have shape (224, 224, 3)

    Returns: the keras model
    """
    resYlY = Keras.Input(shape=(224, 224, 3))
    krnl = Keras.initializers.he_normal(seed=None)
    padding = "same"

    C1 = Keras.layers.Conv2D(
        filters=64,
        kernel_size=(7, 7),
        strides=(2, 2),
        padding=padding,
        activation='relu',
        use_bias=True,
        kernel_initializer=krnl
    )(resYlY)
    PL_C1 = Keras.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding=padding
    )(C1)
    C2 = Keras.layers.Conv2D(
        filters=64,
        kernel_size=(1, 1),
        activation='relu',
        kernel_initializer=krnl
    )(PL_C1)
    C3 = Keras.layers.Conv2D(
        filters=192,
        kernel_size=(3, 3),
        padding=padding,
        activation='relu',
        kernel_initializer=krnl
    )(C2)
    PL_C2 = Keras.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding=padding
    )(C3)
    C31 = inception_block(PL_C2, [64, 96, 128, 16, 32, 32])
    C32 = inception_block(C31, [128, 128, 192, 32, 96, 64])
    PL_C3 = Keras.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding=padding
    )(C32)
    INC_3 = inception_block(PL_C3, [192, 96, 208, 16, 48, 64])
    C4 = inception_block(INC_3, [160, 112, 224, 24, 64, 64])
    C5 = inception_block(C4, [128, 128, 256, 24, 64, 64])
    C6 = inception_block(C5, [112, 144, 288, 32, 64, 64])
    C7 = inception_block(C6, [256, 160, 320, 32, 128, 128])
    PL_C4 = Keras.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding=padding
    )(C7)
    C8 = inception_block(PL_C4, [256, 160, 320, 32, 128, 128])
    C9 = inception_block(C8, [384, 192, 384, 48, 128, 128])
    moyenne = Keras.layers.AveragePooling2D(pool_size=(7, 7),
                                            strides=(1, 1))(C9)
    fn = Keras.layers.Dropout(0.4)(moyenne)
    XX = Keras.layers.Dense(
        units=1000,
        activation='softmax',
        kernel_initializer=krnl
    )(fn)
    resultat = Keras.models.Model(inputs=resYlY, outputs=XX)
    return resultat
