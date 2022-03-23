#!/usr/bin/env python3
""" data Augmentation """

import tensorflow as tf
import numpy as np


def pca_color(image, alphas):
    """
    performs PCA color augmentation as described in the AlexNet paper

    image(tf.Tensor): the image to change
    alphas(tuple): the amount that each channel should change

    Returns
        the augmented image
    """
    img = tf.keras.preprocessing.image.img_to_array(image)
    orig_img = img.astype(float).copy()

    img /= 255.0

    reshape_img = img.reshape(-1, 3)

    eig_vals, eig_vecs = np.linalg.eig(
        np.cov(
            # centered image matrix
            reshape_img - np.mean(reshape_img, axis=0),
            rowvar=False
        )
    )

    eig_vals[::-1].sort()
    eig_vecs = eig_vecs[:, eig_vals[::-1].argsort()]


    mas2 = np.zeros((3, 1))
    mas2[:, 0] = alphas * eig_vals[:]

    # project the image on the eigenvectors
    vector = np.matrix(np.column_stack((eig_vecs))) * np.matrix(mas2)

    for i in range(3):
        # add the mean back
        orig_img[..., i] += vector[i]


    return np.clip(orig_img, 0.0, 255.0).astype(np.uint8)
