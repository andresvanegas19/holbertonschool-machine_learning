#!/usr/bin/env python3
""" Attention """
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """ create an encoder block for a transformer """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        initialized the variables

        Arg:
            dm: the dimensionality of the model
            h: the number of heads
            hidden: the number of hidden units in the fully connected layer
            drop_rate: the dropout rate
        """
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, training, mask=None):
        """
        Public instance method

        Arg:
            x: tensor of shape (batch, input_seq_len, dm)containing the input
                    to the encoder block
            training: boolean to determine if the model is training
            mask: the mask to be applied for multi head attention

        Return:
            tensor of shape (batch, input_seq_len, dm) with the block’s output
        """
        attention_out, _ = self.mha(x, x, x, mask)
        attention_out = self.dropout1(attention_out, training=training)
        first_out = self.layernorm1(x + attention_out)
        ffn_output = self.dense_hidden(first_out)
        ffn_output = self.dense_output(ffn_output)
        ffn_output = self.dropout2(ffn_output, training=training)

        return self.layernorm2(first_out + ffn_output)
