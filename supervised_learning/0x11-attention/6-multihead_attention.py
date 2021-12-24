#!/usr/bin/env python3
""" Attention """
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """ perform multi head attention """

    def __init__(self, dm, h):
        """ Class constructor """
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = int(self.dm // self.h)
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, m, batch):
        """
        split last dim shape transpose result shape

        Args:
            m: tensor shape (batch, seq_len, dm)
            batch: size of batch

        Returns:
            tensor shape (batch, h, seq_len, depth)
        """
        m = tf.reshape(m, (batch, -1, self.h, self.depth))

        return tf.transpose(m, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """
        Public Instance Method

        Q: tensor shape (batch, seq_len_q, dk) contains input to
            generate the query matrix
        K: tensor shape (batch, seq_len_v, dk) contains input to
            generate the key matrix
        V: tensor shape (batch, seq_len_v, dv) contains input to
            generate the value matrix

        Returns:
            output: tensor with last two dims (..., seq_len_q, dm)
                contains scaled dot product attention
            w: tensor with last three dims
                (..., h, seq_len_q, seq_len_v) contains attention w
        """
        batch = tf.shape(K)[0]
        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)

        Q = self.splitHeads(Q, batch)
        K = self.splitHeads(K, batch)
        V = self.splitHeads(V, batch)

        output, w = sdp_attention(Q, K, V, mask)

        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch, -1, self.dm))
        output = self.linear(output)

        return output, w
