#!/usr/bin/env python3
""" performs a valid convolution  """
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    performs a valid convolution on grayscale images:

    images is a numpy.ndarray with shape (m, h, w)
    containing multiple grayscale images
        - m is the number of images
        - h is the height in pixels of the images
        - w is the width in pixels of the images
    kernel is a numpy.ndarray with shape (kh, kw)
    containing the kernel for the convolution
        - kh is the height of the kernel
        - kw is the width of the kernel
    """
    M = images.shape[0]
    H = images.shape[1]
    W = images.shape[2]

    KW = kernel.shape[1]
    kh = kernel.shape[0]

    c_H = int(H - kh + 1)
    c_W = int(W - KW + 1)

    # initialize convolution conv_output tensor
    conv_output = np.zeros((M, c_H, c_W))

    for x in range(c_W):
        for y in range(c_H):

            conv_output[:, y, x] = (
                kernel * images[:, y: y + kh, x: x + KW]
            ).sum(axis=(1, 2))

    return conv_output
