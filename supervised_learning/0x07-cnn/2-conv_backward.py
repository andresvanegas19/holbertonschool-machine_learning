#!/usr/bin/env python3
""" Convolutional neural network """

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    performs back propagation over a convolutional layer of a neural network:

    dZ is a numpy.ndarray of shape (m, h_new, w_new, c_new)
    containing the partial derivatives with respect to the unactivated output
        - of the convolutional layer
        - m is the number of examples
        - h_new is the height of the output
        - w_new is the width of the output
        - c_new is the number of channels in the output
    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
    containing the output of the previous layer
        - h_prev is the height of the previous layer
        - w_prev is the width of the previous layer
        - c_prev is the number of channels in the previous layer
    W is a numpy.ndarray of shape (kh, kw, c_prev, c_new) containing
    the kernels for the convolution
        - kh is the filter height
        - kw is the filter width
    b is a numpy.ndarray of shape (1, 1, 1, c_new) containing
    the biases applied to the convolution
    padding is a string that is either same or valid,
    indicating the type of padding used
    stride is a tuple of (sh, sw) containing the strides for the convolution
        - sh is the stride for the height
        - sw is the stride for the width

    Returns: the partial derivatives with respect to the previous
    layer (dA_prev), the kernels (dW), and the biases (db), respectively
    """

    M = dZ.shape[0]
    H = dZ.shape[1]
    W_N = dZ.shape[2]
    C = dZ.shape[3]

    H_P, W_P, _ = A_prev.shape[1:]
    kh, kw = W.shape[:2]

    sh = stride[0]
    sw = stride[1]

    ph, pw = 0, 0
    dW = np.zeros(W.shape)

    if padding == "same":
        ph = int((H * sh - H_P + kh - 1) / 2)
        pw = int((W_N * sw - W_P + kw - 1) / 2)

    pad = np.pad(
        A_prev,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        'constant',
        constant_values=0
    )

    dA_p = np.zeros(pad.shape)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    for image in range(M):
        for channel in range(C):
            for row in range(H):
                for col in range(W_N):
                    grad = np.multiply(
                        W[:, :, :, channel],
                        dZ[image, row, col, channel]
                    )
                    dA_p[
                        image,
                        row * sh:row * sh + kh,
                        col * sw:col * sw + kw,
                        :
                    ] += grad
                    grad = np.multiply(
                        pad[
                            image,
                            row * sh:row * sh + kh,
                            col * sw:col * sw + kw,
                            :
                        ],
                        dZ[image, row, col, channel]
                    )
                    dW[:, :, :, channel] += grad

    return  \
        dA_p[
            :,
            ph:dA_p.shape[1] - ph,
            pw:dA_p.shape[2] - pw,
            :
        ], \
        dW,\
        db
