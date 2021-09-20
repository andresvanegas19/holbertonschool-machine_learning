#!/usr/bin/env python3
""" Made a cnn with keras """


import tensorflow.keras as Keras


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Build the model in cnn

    Args:
        nx ([type]): [description]
        layers ([type]): [description]
        activations ([type]): [description]
        lambtha ([type]): [description]
        keep_prob ([type]): [description]

    Returns:
        [type]: [description]
    """
    reg = Keras.regularizers.l2
    model = Keras.Sequential()
    model.add(Keras.layers.Dense(
        layers[0],
        input_shape=(nx,),
        activation=activations[0],
        kernel_regularizer=reg(lambtha))
    )

    for layer, act in zip(layers[1:], activations[1:]):
        model.add(Keras.layers.Dropout(1 - keep_prob))
        model.add(
            Keras.layers.Dense(
                layer,
                activation=act,
                kernel_regularizer=reg(lambtha)
            )
        )

    return model
