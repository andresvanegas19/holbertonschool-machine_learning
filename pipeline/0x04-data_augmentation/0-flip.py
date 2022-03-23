#!/usr/bin/env python3
""" data Augmentation """

import tensorflow as tf


def flip_image(image):
    """
    flips an image horizontally

    Args:
        image (tf.Tensor): the image to flip

    Returns
        the flipped image
    """

    return tf.image.flip_left_right(image)