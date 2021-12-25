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
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def s_head(self, x, batch):
        """
        split last dim shape transpose result shape

        Args:
            m: tensor shape (batch, seq_len, dm)
            batch: size of batch

        Returns:
            tensor shape (batch, h, seq_len, depth)
        """
        x = tf.reshape(x, (batch, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

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
        batch = tf.shape(Q)[0]

        q = self.Wq(Q)
        k = self.Wk(K)
        v = self.Wv(V)
        q = self.s_head(q, batch)
        k = self.s_head(k, batch)
        v = self.s_head(v, batch)

        scaled, wights_atten = sdp_attention(q, k, v, mask)
        scaled = tf.transpose(
            scaled,
            perm=[0, 2, 1, 3]
        )

        c_attetion = tf.reshape(
            scaled,
            (batch, -1, self.dm)
        )
        output = self.linear(c_attetion)

        return output, wights_atten
