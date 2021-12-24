#!/usr/bin/env python3
""" Attention """
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """
    calculates the scaled dot product attention

    Args:
        Q(tensor): containing the query matrix
        K(tensor): containing the key matrix
        V(tensor): containing the value matrix
        mask(tensor): containing the optional mask, or defaulted to None

    Returns:
        outputa: tensor with its last two dimensions as (..., seq_len_q, dv)
            containing the scaled dot product attention
        weights: a tensor with its last two dimensions as (..., seq_len_q,
            seq_len_v) containing the attention weights
    """

    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_q = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(dk)

    if mask is not None:
        scaled_q += (mask * -1e9)

    weights = tf.nn.softmax(scaled_q, axis=-1)

    return tf.matmul(weights, V), weights
