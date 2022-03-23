#!/usr/bin/env python3
""" data Augmentation """

import tensorflow as tf


def shear_image(image, intensity):
    """
    randomly shears an image

    image(tf.Tensor): the image to shear
    intensity: is the intensity with which the image should be sheared

    Returns:
        the sheared image
    """
    img = tf.keras.preprocessing.image.img_to_array(image)
    sheared_img = tf.keras.preprocessing.image.random_shear(
        img,
        intensity,
        row_axis=0,
        col_axis=1,
        channel_axis=2
    )

    return tf.keras.preprocessing.image.array_to_img(sheared_img)