#!/usr/bin/env python3
""" performs a valid convolution """
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    performs pooling on images:

    images is a numpy.ndarray with shape (m, h, w, c)
    containing multiple images
        - m is the number of images
        - h is the height in pixels of the images
        - w is the width in pixels of the images
        - c is the number of channels in the image
    kernel_shape is a tuple of (kh, kw)
    containing the kernel shape for the pooling
        - kh is the height of the kernel
        - kw is the width of the kernel
    stride is a tuple of (sh, sw)
        - sh is the stride for the height of the image
        - sw is the stride for the width of the image
    mode indicates the type of pooling
        - max indicates max pooling
        - avg indicates average pooling

    Returns: a numpy.ndarray containing the pooled images
    """

    N = images.shape[0]
    HI = images.shape[1]  # image height
    WI = images.shape[2]  # image width
    channels = images.shape[3]

    HK = kernel_shape[0]  # hight kernel
    WK = kernel_shape[1]

    sh = stride[0]
    sw = stride[1]

    height = int((HI - HK) // sh + 1)
    width = int((WI - WK) // sw + 1)

    search = np.zeros((
        N,
        height,
        width,
        channels
    ))

    for space in range(width):
        for jmp in range(height):
            image_ = images[
                :,
                (sh * jmp):(sh * jmp) + HK,
                (sw * space):(sw * space) + WK
            ]

            if mode != 'max':
                search[:, jmp, space] = np.average(image_, axis=(1, 2))
            else:
                search[:, jmp, space] = np.max(image_, axis=(1, 2))

    return search
