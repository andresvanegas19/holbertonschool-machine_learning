#!/usr/bin/env python3
""" Attention """
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    """ create the decoder for a transformer """

    def __init__(self, N, dm, h, hidden, input_vocab, max_seq_len, drop_rate=0.1):
        """
        Class constructor

        Args:
            N: the number of blocks in the encoder
            dm: the dimensionality of the model
            h: the number of heads
            hidden: the number of hidden units in the fully connected layer
            target_vocab: the size of the target vocabulary
            max_seq_len: the maximum sequence length possible
            drop_rate: the dropout rate
        """
        super(Encoder, self).__init__()
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, self.dm)
        self.dropout = tf.keras.layers.Dropout(drop_rate)
        self.N = N
        self.blocks = [
            EncoderBlock(dm, h, hidden, drop_rate) for _ in range(N)
        ]

    def call(self, x, training, mask):
        """
        Public instance method

        Args:
            x: a tensor of shape (batch, target_seq_len, dm)
                containing the input to the decoder
            encoder_output: a tensor of shape (batch, input_seq_len, dm)
                containing the output of the encoder
            training: a boolean to determine if the model is training
            look_ahead_mask: the mask to be applied to the first multi
                head attention layer
            padding_mask: the mask to be applied to the second multi
                head attention layer

        Returns:
            a tensor of shape (batch, target_seq_len, dm)
            containing the decoder output
        """
        seq_len = x.shape[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len]
        res = self.dropout(x, training=training)

        for block in self.blocks:
            res = block(res, training, mask)

        return res
