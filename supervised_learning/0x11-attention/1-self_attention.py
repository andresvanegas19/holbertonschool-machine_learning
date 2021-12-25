#!/usr/bin/env python3
""" Attention """
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """ calculate the attention for machine """

    def __init__(self, units):
        """
        Class constructor

        Args:
            units (integer): representing the number of hidden units
            in the alignment model
        """
        super().__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """
        Public instance method

        Args:
            s_prev (tensor of shape): the previous decoder hidden state
            hidden_states ([type]): [description]
        """
        new_s_prev = tf.expand_dims(s_prev, axis=1)
        score = self.V(tf.nn.tanh(self.W(new_s_prev) + self.U(hidden_states)))
        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(weights * hidden_states, axis=1)

        return context, weights
