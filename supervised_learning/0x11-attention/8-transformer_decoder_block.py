#!/usr/bin/env python3
""" Attention """
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """ create an encoder block for a transformer """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Class constructor

        Arg:
            dm: the dimensionality of the model
            h: the number of heads
            hidden: the number of hidden units in the fully connected layer
            drop_rate: the dropout rate
        """
        super(DecoderBlock, self).__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(
            units=hidden,
            activation='relu'
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dense_output = tf.keras.layers.Dense(units=dm)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Public instance method

        x: a tensor of shape (batch, target_seq_len, dm)containing the
            input to the decoder block
        encoder_output: a tensor of shape (batch, input_seq_len, dm)
            containing the output of the encoder
        training: a boolean to determine if the model is training
        look_ahead_mask: the mask to be applied to the first multi
            head attention layer
        padding_mask: the mask to be applied to the second multi
            head attention layer

        Returns:
            a tensor of shape (batch, target_seq_len, dm)
            containing the block’s output
        """
        # first attention block
        a, _ = self.mha1(x, x, x, mask=look_ahead_mask)
        b = self.dropout1(a, training=training)
        # c = self.layernorm1(x + b)
        c = self.layernorm1(b + x)
        # decoder block
        d, _ = self.mha2(c, encoder_output, encoder_output, mask=padding_mask)
        # if self.dropout2 is not None:
        e = self.dropout2(d, training=training)
        f = self.layernorm2(e + c)
        # Implement the second FFN
        g = self.dense_hidden(f)
        # g = self.dense_output(g)
        h = self.dense_output(g)

        return self.layernorm2(
            self.dropout2(h, training=training) + f
        )
