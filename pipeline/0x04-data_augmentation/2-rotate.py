#!/usr/bin/env python3
""" data Augmentation """

import tensorflow as tf


def rotate_image(image):
    """
    rotates an image by 90 degrees counter-clockwise:

    image (tf.Tensor): the image to rotate

    Returns:
        the rotated image
    """
    rotate = tf.image.rot90(image, k=1)

    return (rotate)