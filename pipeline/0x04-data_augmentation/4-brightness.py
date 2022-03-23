#!/usr/bin/env python3
""" data Augmentation """

import tensorflow as tf


def change_brightness(image, max_delta):
    """
    randomly changes the brightness of an image

    Args:
        image (tf.Tensor): the image to flip
        max_delta: maximum amount the image should be brightened (or darkened)

    Returns
        the sheared image
    """

    return tf.image.random_brightness(image, max_delta)
