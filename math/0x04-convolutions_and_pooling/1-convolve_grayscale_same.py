#!/usr/bin/env python3
""" performs a valid convolution  """
import numpy as np


def convolve_grayscale_same(images, kernel):
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
    convoluted = np.zeros((M, H, W))

    KH = kernel.shape[0]
    KW = kernel.shape[1]
    ph = int(KH / 2)
    pw = int(KW / 2)

    pad = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw)),
        'constant'
    )

    for h in range(H):
        # hight
        for w in range(W):
            # large
            convoluted[:, h, w] = (
                # image slide
                kernel * pad[:, h: h + KH, w: w + KW]
            ).sum(axis=(1, 2))

    return convoluted
