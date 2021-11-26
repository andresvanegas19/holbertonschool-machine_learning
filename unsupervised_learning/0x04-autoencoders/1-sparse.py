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
        - auto is the sparse autoencoder model
    """

    input_encoder = keras.Input(shape=(input_dims,))
    act_reg = keras.regularizers.l1(lambtha)
    output = keras.layers.Dense(
        hidden_layers[0],
        activation='relu',
        activity_regularizer=act_reg
    )(input_encoder)

    for i in range(1, len(hidden_layers)):
        output = keras.layers.Dense(
            hidden_layers[i],
            activation='relu',
            activity_regularizer=act_reg
        )(output)

    latent = keras.layers.Dense(
        latent_dims,
        activation='relu',
        activity_regularizer=act_reg
    )(output)
    encoder = keras.models.Model(
        inputs=input_encoder,
        outputs=latent
    )

    #  encoder goes backwards in dimension till input_dim
    input_decoder = keras.Input(shape=(latent_dims,))
    o_two = keras.layers.Dense(
        hidden_layers[-1],
        activation='relu'
    )(input_decoder)

    for i in range(len(hidden_layers) - 2, -1, -1):
        o_two = keras.layers.Dense(
            hidden_layers[i],
            activation='relu'
        )(o_two)
    # decoder goes backwards in dimension till input_dim
    last_layer = keras.layers.Dense(input_dims, activation='sigmoid')(o_two)

    decoder = keras.models.Model(inputs=input_decoder, outputs=last_layer)
    input_auto = keras.Input(shape=(input_dims,))
    encoder_out = encoder(input_auto)
    auto = keras.models.Model(inputs=input_auto, outputs=decoder(encoder_out))
    auto.compile(loss='binary_crossentropy', optimizer='Adam')

    return encoder, decoder, auto
