#!/usr/bin/env python3
""" data Augmentation """

import tensorflow as tf


def change_hue(image, delta):
    """
    changes the hue of an image

    image(tf.Tensor): the image to change
    delta: is the amount the hue should change

    Returns
        the altered image
    """

    return tf.image.adjust_hue(image, delta)