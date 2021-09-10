#!/usr/bin/env python3
""" Convolutional neural network """
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    performs forward propagation over a pooling layer of a neural network:

    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
    containing the output of the previous layer
        - m is the number of examples
        - h_prev is the height of the previous layer
        - w_prev is the width of the previous layer
        - c_prev is the number of channels in the previous layer
    kernel_shape is a tuple of (kh, kw) containing the size of the
    kernel for the pooling
        - kh is the kernel height
        - kw is the kernel width
    stride is a tuple of (sh, sw) containing the strides for the pooling
        - sh is the stride for the height
        - sw is the stride for the width
    mode is a string containing either max or avg, indicating whether
    to perform maximum or average pooling, respectively

    Returns: the output of the pooling layer
    """

    M = A_prev.shape[0]
    H = A_prev.shape[1]
    W = A_prev.shape[2]
    C = A_prev.shape[3]

    kh = kernel_shape[0]  # kernel
    kw = kernel_shape[1]

    sh = stride[0]
    sw = stride[1]

    ph = (H - kh) // sh + 1
    pw = (W - kw) // sw + 1

    convolutional = np.zeros((M, ph, pw, C))

    for row in range(ph):
        for col in range(pw):
            slicing = A_prev[
                :,
                row * sh:row * sh + kh,
                col * sw:col * sw + kh
            ]

            if mode == "avg":
                convolutional[:, row, col] = slicing.mean(axis=(1, 2))
            if mode == "max":
                convolutional[:, row, col] = slicing.max(axis=(1, 2))

    return convolutional
