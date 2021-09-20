#!/usr/bin/env python3
"""
You must use one of the applications listed
in Keras Applications
Your script must save your trained model in
the current working directory as cifar10.h5
Your saved model should be compiled
Your saved model should have a validation
accuracy of 87% or higher
Your script should not run when the file is imported
Hint1: The training and tweaking of hyperparameters 
may take a while so start early!
Hint2: The CIFAR 10 dataset contains 32x32 pixel
images, however most of the Keras applications are
trained on much larger images. Your first layer
should be a lambda layer that scales up the data to the correct size
Hint3: You will want to freeze most of the
application layers. Since these layers will
always produce the same output, you should
compute the output of the frozen layers ONCE
and use those values as input to train the
remaining trainable layers. This will save you A LOT of time.
"""
import tensorflow.keras as K
import numpy as np


def preprocess_data(X, Y):
    """[summary]
    pre-processes the data for your model:

    X is a numpy.ndarray of shape (m, 32, 32, 3)
    containing the CIFAR 10 data, where m is the number of data points
    Y is a numpy.ndarray of shape (m,) containing the CIFAR 10 labels for X
    Returns: X_p, Y_p
    X_p is a numpy.ndarray containing the preprocessed X
    Y_p is a numpy.ndarray containing the preprocessed Y

    Args:
        X ([type]): [description]
        Y ([type]): [description]

    Returns:
        [type]: [description]
    """
    return \
        K.applications.densenet.preprocess_input(X), \
        K.utils.to_categorical(Y, 10)


def learning_rate(epoch):
    """
    modificate the date rate limiting
    """
    return 0.001 / (1 + 1 * epoch)


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()

    x_train = K.applications.densenet.preprocess_input(x_train)
    y_train = K.utils.to_categorical(y_train, 10)

    g_train = K.preprocessing.image.ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    ).flow(x_train, y_train, batch_size=30)

    x_test = K.applications.densenet.preprocess_input(x_test)
    y_test = K.utils.to_categorical(y_test, 10)

    inputs = K.Input(shape=(32, 32, 3))
    inputs = K.layers.UpSampling2D()(inputs)

    network = K.applications.densenet.DenseNet121(
        include_top=False,
        pooling='max',
        input_tensor=inputs,
        weights='imagenet'
    )

    output = network.layers[-1].output
    output = K.layers.Flatten()(output)
    output = K.layers.Dense(512, activation='relu')(output)
    output = K.layers.Dropout(0.15)(output)
    output = K.layers.Dense(256, activation='relu')(output)
    output = K.layers.Dropout(0.15)(output)
    output = K.layers.Dense(10, activation='softmax')(output)

    model = K.models.Model(network.input, output)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['acc']
    )

    # callbacks configuration
    callbacks = []

    # learning rate
    lrr__callback = K.callbacks.LearningRateScheduler(
        learning_rate,
        verbose=1
    )
    callbacks.append(lrr__callback)

    es_callback = K.callbacks.EarlyStopping(
        monitor='val_acc',
        mode='max',
        verbose=1,
        patience=5
    )
    callbacks.append(es_callback)

    mc__callback = K.callbacks.ModelCheckpoint(
        'cifar10.h5',
        monitor='val_acc',
        mode='max',
        verbose=1,
        save_best_only=True
    )
    callbacks.append(mc__callback)

    history = model.fit(
        g_train,
        validation_data=(x_test, y_test),
        batch_size=128,
        callbacks=callbacks,
        epochs=32,
        verbose=1
    )
