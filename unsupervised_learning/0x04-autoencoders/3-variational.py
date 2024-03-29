#!/usr/bin/env python3
"""" autoencoders  """

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    creates a variational autoencoder
    Args:
        input_dims - is an integer containing the dimensions
            of the model input
        hidden_layers - is a list containing the number of
            nodes for each hidden layer in the encoder, respectively
            the hidden layers should be reversed for the decoder
        latent_dims - is an integer containing the dimensions
            of the latent space representation
    Returns:
        encoder - is the encoder model
        decoder -  is the decoder model
        auto - is the full autoencoder model
    """
    inp_enc = keras.layers.Input(shape=(input_dims,))
    input_encoded = inp_enc

    for n_h in hidden_layers:
        encoded = keras.layers.Dense(n_h, activation='relu')(input_encoded)
        input_encoded = encoded

    latent = keras.layers.Dense(latent_dims, activation='relu')(encoded)
    encoder = keras.models.Model(inp_enc, latent)
    input_decoder = keras.layers.Input(shape=(latent_dims,))
    input_decoded = input_decoder

    for _, n in enumerate(hidden_layers[::-1]):
        activation = 'relu'
        decoded = keras.layers.Dense(n, activation=activation)(input_decoded)
        input_decoded = decoded

    decoded = keras.layers.Dense(
        input_dims, activation='sigmoid'
    )(input_decoded)
    decoder = keras.models.Model(input_decoder, decoded)

    input_auto = keras.layers.Input(shape=(input_dims,))
    encoder_out = encoder(input_auto)
    auto = keras.models.Model(input_auto, decoder(encoder_out))
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
