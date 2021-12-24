#!/usr/bin/env python3
""" Attention """
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """ decode for machine translation """

    def __init__(self, vocab, embedding, units, batch):
        """
        Class constructor

        Args:
            vocab (integer): the size of the output vocabulary
            embedding (integer): the dimensionality of the embedding vector
            units (integer): number of hidden units in the RNN cell
            batch (integer): batch size
        """
        super(RNNDecoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units,
            recurrent_initializer='glorot_uniform',
            return_sequences=True,
            return_state=True
        )
        self.F = tf.keras.layers.Dense(vocab)

    def call(self, x, s_prev, hidden_states):
        """
        Public instance method

        Args:
            x (tensor of shape):  containing the previous word in the target
                sequence as an index of the target vocabulary
            s_prev (tensor of shape): (batch, units) containing the previous
                decoder hidden state
            hidden_states (tensor of shape): (batch, input_seq_len, units)
                containing the outputs of the encoder

        Returns:
            y: the output word as a one hot vector in the target vocabulary
            s: the new decoder hidden state
        """
        vec_cont, _ = self.attention(s_prev, hidden_states)
        x = tf.concat(
            [
                tf.expand_dims(vec_cont, 1),
                self.embedding(x)
            ],
            axis=-1
        )
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        y = self.F(output)

        return y, state
