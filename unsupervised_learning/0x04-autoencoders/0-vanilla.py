#!/usr/bin/env python3
"""" autoencoders  """

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    creates an autoencoder

    - The autoencoder model should be compiled using adam optimization
    and binary cross-entropy loss
    - All layers should use a relu activation except for the last layer in
    the decoder, which should use sigmoid

    Args:
        input_dims: is an integer containing the dimensions of the model input
        hidden_layers: a list containing the number of nodes
        in each hidden layer
        latent_dims: dimensions of the latent space

    Returns:
        encoder is the encoder model
        decoder is the decoder model
        auto is the full autoencoder model
    """

    enco_inpt = keras.layers.Input(shape=(input_dims,))
    prev = enco_inpt

    # Encoder
    for i in hidden_layers:
        tmp = keras.layers.Dense(i, activation='relu')(prev)
        prev = tmp

    # Latent space
    encoder = keras.models.Model(
        enco_inpt,
        # bottleneck
        keras.layers.Dense(latent_dims, activation='relu')(prev)
    )

    # Decoder
    decoder_input = keras.layers.Input(shape=(latent_dims,))
    prev = decoder_input

    for i in hidden_layers[::-1]:
        tmp = keras.layers.Dense(i, activation='relu')(prev)
        prev = tmp

    decoder = keras.models.Model(
        decoder_input,
        # output_layer
        keras.layers.Dense(input_dims, activation='sigmoid')(prev)
    )

    # Autoencoder
    input_layer = keras.layers.Input(shape=(input_dims,))
    encoder_out = encoder(input_layer)
    auto = keras.models.Model(
        input_layer,
        # decoder_out
        decoder(encoder_out)
    )
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
