#!/usr/bin/env python3
""" Convolutional neural network """


import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    performs back propagation over a pooling layer of a neural network:

    dA is a numpy.ndarray of shape (m, h_new, w_new, c_new) containing
    the partial derivatives with respect to the output of the pooling layer
        - m is the number of examples
        - h_new is the height of the output
        - w_new is the width of the output
        - c is the number of channels
    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c)
    containing the output of the previous layer
        - h_prev is the height of the previous layer
        - w_prev is the width of the previous layer
    kernel_shape is a tuple of (kh, kw) containing the size of
    the kernel for the pooling
        - kh is the kernel height
        - kw is the kernel width
    stride is a tuple of (sh, sw) containing the strides for the pooling
        - sh is the stride for the height
        - sw is the stride for the width
    mode is a string containing either max or avg, indicating whether to
    perform maximum or average pooling, respectively

    Returns: the partial derivatives with respect to the previous layer

    """
    M = dA.shape[0]
    H = dA.shape[1]
    W = dA.shape[2]
    C = dA.shape[3]

    kh = kernel_shape[0]  # kernel
    kw = kernel_shape[1]

    sh = stride[0]
    sw = stride[1]

    convuled = np.zeros(A_prev.shape)

    for image in range(M):
        for channel in range(C):
            for row in range(H):
                for col in range(W):

                    if mode == "max":
                        pool = A_prev[
                            image,
                            row * sh:row * sh + kh,
                            col * sw:col * sw + kw,
                            channel
                        ]

                        mask = pool == np.max(pool)
                        convuled[
                            image,
                            row * sh:row * sh + kh,
                            col * sw:col * sw + kw,
                            channel
                        ] += dA[image, row, col, channel] * mask

                    # mode is avg
                    else:
                        convuled[
                            image,
                            row * sh:row * sh + kh,
                            col * sw:col * sw + kw,
                            channel
                        ] += dA[image, row, col, channel] / (kh * kw)
                        # derivate
    return convuled
