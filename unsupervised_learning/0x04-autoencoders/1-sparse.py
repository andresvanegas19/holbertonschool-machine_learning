#!/usr/bin/env python3
"""" autoencoders  """


import tensorflow.keras as keras


def sparse(input_dims, hidden_layers, latent_dims, lambtha):
    """
    creates a sparse autoencoder

    Args:
        input_dims: integer containing the dimensions of the model input
        hidden_layers: list containing the number of nodes for each hidden
            layer in the encoder, the hidden layers should be reversed
            for the decoder

        latent_dims [integer]: containing the dimensions of the latent
            space representation
        lambtha: regularization parameter used for L1 regularization
            on the encoded output

    Returns:
        - encoder is the encoder model
        - decoder is the decoder model
        - autoenc is the sparse autoencoder model
    """
    inp_enc = keras.layers.Input(shape=(input_dims,))
    inp_dec = inp_enc

    regularizer = keras.regularizers.l1(lambtha)
    for h_n in hidden_layers:
        encoded = keras.layers.Dense(
            h_n,
            activation='relu',
            activity_regularizer=regularizer
        )(inp_dec)
        inp_dec = encoded

    latent = keras.layers.Dense(
        latent_dims,
        activation='relu',
        activity_regularizer=regularizer
    )(encoded)
    encoder = keras.models.Model(inp_enc, latent)

    input_decoder = keras.layers.Input(shape=(latent_dims,))
    input_decoded = input_decoder
    for _, n in enumerate(hidden_layers[::-1]):
        activation = 'relu'
        decoded = keras.layers.Dense(n, activation=activation)(input_decoded)
        input_decoded = decoded
    decoded = keras.layers.Dense(
        input_dims,
        activation='sigmoid'
    )(input_decoded)
    decoder = keras.models.Model(input_decoder, decoded)

    input_auto = keras.layers.Input(shape=(input_dims,))
    encoder_out = encoder(input_auto)
    autoenc = keras.models.Model(input_auto, decoder(encoder_out))
    autoenc.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, autoenc
