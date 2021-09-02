#!/usr/bin/env python3
""" performs a valid convolution """
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
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
    padding is a tuple of (ph, pw)
        - ph is the padding for the height of the image
        - pw is the padding for the width of the image

    Returns: a numpy.ndarray containing the convolved images
    """

    M = images.shape[0]
    H = images.shape[1]
    W = images.shape[2]

    KH = kernel.shape[0]
    KW = kernel.shape[1]
    ph, pw = padding

    pad = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw)),
        'constant'
    )

    convoluted = np.zeros((
        M,
        H + 2 * ph - KH + 1,  # CH
        W + 2 * pw - KW + 1  # CW
    ))

    # im = np.arange(M) # Falling add another dimension
    hei = H + 2 * ph - KH + 1
    cew = W + 2 * pw - KW + 1

    for h in range(hei):
        for w in range(cew):
            convoluted[:, h, w] = (
                kernel * pad[:, h:h + KH, w:w + KW]  # Get the med
            ).sum(axis=(1, 2))

    return convoluted
