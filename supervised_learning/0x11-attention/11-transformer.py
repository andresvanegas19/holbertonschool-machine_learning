#!/usr/bin/env python3
""" Attention """
import tensorflow as tf
Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


class Transformer(tf.keras.Model):
    """ create the decoder for a transformer """

    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        """ init the transformer """
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            N,
            dm,
            h,
            hidden,
            input_vocab,
            max_seq_input,
            drop_rate
        )
        self.decoder = Decoder(
            N,
            dm,
            h,
            hidden,
            target_vocab,
            max_seq_target,
            drop_rate
        )
        self.linear = tf.keras.layers.Dense(units=target_vocab)

    def call(self, inputs, target, training, encoder_mask,
             look_ahead_mask, decoder_mask):
        """
            Public instance method

            x: a tensor of shape (batch, target_seq_len, dm)
                containing the input to the decoder
            encoder_output: a tensor of shape (batch, input_seq_len, dm)
                containing the output of the encoder
            training: a boolean to determine if the model is training
            look_ahead_mask: the mask to be applied to the first
                multi head attention layer
            padding_mask: the mask to be applied to the second
                multi head attention layer

            Returns:
                a tensor of shape (batch, target_seq_len, dm) containing
                the decoder output
        """
        encoder_output = self.encoder(inputs, training, encoder_mask)
        decoder_output = self.decoder(
            target,
            encoder_output,
            training,
            look_ahead_mask,
            decoder_mask
        )

        return self.linear(decoder_output)
