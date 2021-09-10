#!/usr/bin/env python3
""" Convolutional neural network """

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    performs forward propagation over a convolutional
    layer of a neural network:

    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
    containing the output of the previous layer
        - m is the number of examples
        - h_prev is the height of the previous layer
        - w_prev is the width of the previous layer
        - c_prev is the number of channels in the previous layer
    W is a numpy.ndarray of shape (kh, kw, c_prev, c_new) containing
    the kernels for the convolution
        - kh is the filter height
        - kw is the filter width
        - c_prev is the number of channels in the previous layer
        - c_new is the number of channels in the output
    b is a numpy.ndarray of shape (1, 1, 1, c_new)
    containing the biases applied to the convolution
    activation is an activation function applied to the convolution
    padding is a string that is either same or valid,
    indicating the type of padding used
    stride is a tuple of (sh, sw) containing the strides for the convolution
        - sh is the stride for the height
        - sw is the stride for the width

    Returns: the output of the convolutional layer
    """

    m = A_prev.shape[0]
    h_prev = A_prev.shape[1]
    w_prev = A_prev.shape[2]

    kh = W.shape[0]  # kernel
    kw = W.shape[1]
    c_new = W.shape[3]

    sh = stride[0]
    sw = stride[1]

    ph, pw = 0, 0

    if padding == 'same':
        ph = int(((h_prev - 1) * sh + kh - h_prev) / 2)
        pw = int(((w_prev - 1) * sw + kw - w_prev) / 2)

    if isinstance(padding, tuple):
        ph, pw = padding

    pad = np.pad(
        A_prev,
        pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant',
        constant_values=0
    )

    W_c = int(((w_prev + 2 * pw - kw) / sw)) + 1
    H_c = int(((h_prev + 2 * ph - kh) / sh)) + 1

    prev_convultion = np.zeros((m, H_c, W_c, c_new))

    for j in range(H_c):
        for k in range(W_c):
            for n in range(c_new):

                prev_convultion[:, j, k, n] = np.multiply(
                    pad[
                        :,
                        j * sh:j * sh + kh,
                        k * sw:k * sw + kw
                    ],
                    W[:, :, :, n]
                ).sum(axis=(1, 2, 3))

    return activation(prev_convultion + b)  # Z
