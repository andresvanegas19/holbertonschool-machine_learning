#!/usr/bin/env python3
""" Convolutional neural network """
import tensorflow as tf


def lenet5(x, y):
    """
    builds a modified version of the LeNet-5 architecture using tensorflow:

    x is a tf.placeholder of shape (m, 28, 28, 1)
    containing the input images for the network
        - m is the number of images
    y is a tf.placeholder of shape (m, 10) containing the
    one-hot labels for the network
    The model should consist of the following layers in order:
        - Convolutional layer with 6 kernels of shape 5x5 with same padding
        - Max pooling layer with kernels of shape 2x2 with 2x2 strides
        - Convolutional layer with 16 kernels of shape 5x5 with valid padding
        - Max pooling layer with kernels of shape 2x2 with 2x2 strides
        - Fully connected layer with 120 nodes
        - Fully connected layer with 84 nodes
        - Fully connected softmax output layer with 10 nodes

    TODO: All layers requiring initialization should initialize
    their kernels with the he_normal initialization method:
        TODO: - tf.contrib.layers.variance_scaling_initializer()

    Returns:
        - a tensor for the softmax activated output
        - a training operation that utilizes Adam optimization
        (with default hyperparameters)
        - a tensor for the loss of the netowrk
        - a tensor for the accuracy of the network
    """
    # All layers requiring initialization should initialize
    init = tf.contrib.layers.variance_scaling_initializer()

    activation = tf.nn.relu

    conv2d1 = tf.layers.Conv2D(
        6,
        (5, 5),
        activation=activation,
        padding='same',
        kernel_initializer=init
    )(x)

    con2d2 = tf.layers.Conv2D(
        16,
        (5, 5),
        activation=activation,
        padding='valid',
        kernel_initializer=init
    )(tf.layers.MaxPooling2D((2, 2), (2, 2),)(conv2d1))

    maxpool2 = tf.layers.MaxPooling2D(
        (2, 2),
        (2, 2),
    )(con2d2)

    fullcc1 = tf.layers.Dense(
        120,
        activation=activation,
        kernel_initializer=init
    )(tf.layers.Flatten()(maxpool2))

    fullcc2 = tf.layers.Dense(
        84,
        activation=activation,
        kernel_initializer=init
    )(fullcc1)

    full_layer3 = tf.layers.Dense(
        10,
        kernel_initializer=init
    )(fullcc2)

    prediction = tf.equal(
        tf.argmax(full_layer3, 1),
        tf.argmax(y, 1)
    )

    loss = tf.losses.softmax_cross_entropy(y, full_layer3)

    # output, train loss and accuracy
    return \
        tf.nn.softmax(full_layer3), \
        tf.train.AdamOptimizer().minimize(loss), \
        loss, \
        tf.reduce_mean(tf.cast(prediction, tf.float32))
