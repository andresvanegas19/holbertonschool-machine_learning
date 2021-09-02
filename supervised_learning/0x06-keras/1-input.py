#!/usr/bin/env python3
""" Module for train a deep learning model with keras """
import tensorflow.keras as Keras


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    builds a neural network with the Keras library:

    nx is the number of input features to the network
    layers is a list containing the number of nodes
    in each layer of the network
    activations is a list containing the activation
    functions used for each layer of the network
    lambtha is the L2 regularization parameter
    keep_prob is the probability that a node will be kept for dropout

    Returns: the keras model
    """

    kernel_reg = Keras.regularizers.l2(lambtha)

    for i in range(len(layers)):
        if i == 0:
            input = Keras.Input(shape=(nx,))
            hidden = Keras.layers.Dense(
                layers[i],
                activation=activations[i],
                kernel_regularizer=kernel_reg
            )(input)

        else:
            dropout = Keras.layers.Dropout(1 - keep_prob)(hidden)
            hidden = Keras.layers.Dense(
                layers[i],
                activation=activations[i],
                kernel_regularizer=kernel_reg
            )(dropout)

    return Keras.Model(input, hidden)
