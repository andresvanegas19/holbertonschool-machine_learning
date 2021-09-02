#!/usr/bin/env python3
""" performs a valid convolution """
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    performs a convolution on grayscale images:

    images is a numpy.ndarray with shape (m, h, w)
    containing multiple grayscale images
        - m is the number of images
        - h is the height in pixels of the images
        - w is the width in pixels of the images
    kernel is a numpy.ndarray with shape (kh, kw)
    containing the kernel for the convolution
        - kh is the height of the kernel
        - kw is the width of the kernel
    padding is either a tuple of (ph, pw), ‘same’, or ‘valid’
        if ‘same’, performs a same convolution
        if ‘valid’, performs a valid convolution
        if a tuple:
            - ph is the padding for the height of the image
            - pw is the padding for the width of the image
    stride is a tuple of (sh, sw)
        - sh is the stride for the height of the image
        - sw is the stride for the width of the image

    Returns: a numpy.ndarray containing the convolved images
    """

    M = images.shape[0]
    H = images.shape[1]
    W = images.shape[2]

    KH = kernel.shape[0]
    KW = kernel.shape[1]

    SH = stride[0]
    SW = stride[1]

    ph, pw = 0, 0

    if padding == 'same':
        ph = int(((H - 1) * SH + KH - H) / 2) + 1  # get the c nub
        pw = int(((W - 1) * SW + KW - W) / 2) + 1

    if isinstance(padding, tuple):
        ph, pw = padding

    pad = np.pad(
        images,
        pad_width=((0, 0), (ph, ph), (pw, pw)),
        mode='constant',
        constant_values=0
    )
    W_c = int(((W + 2 * pw - KW) / SW) + 1)
    H_c = int(((H + 2 * ph - KW) / SH) + 1)

    convoluted = np.zeros((M, H_c, W_c))

    for h in range(H_c):
        for w in range(W_c):
            # Use numpy for more faster
            convoluted[:, h, w] = np.multiply(
                pad[
                    :,
                    h * SH: h * SH + KH,
                    w * SW: w * SW + KW
                ], kernel
            ).sum(axis=(1, 2))

    return convoluted
