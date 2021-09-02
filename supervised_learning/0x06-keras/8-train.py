#!/usr/bin/env python3
""" Module for train a deep learning model with keras """
import tensorflow.keras as Keras


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, verbose=True, shuffle=False):
    """
    trains a model using mini-batch gradient descent:

    network is the model to train
    data is a numpy.ndarray of shape (m, nx) containing the input data
    labels is a one-hot numpy.ndarray of shape
    (m, classes) containing the labels of data

    batch_size is the size of the batch used for mini-batch gradient descent
    epochs is the number of passes through data for
    mini-batch gradient descent

    verbose is a boolean that determines if output
    should be printed during training

    shuffle is a boolean that determines whether to shuffle the
    batches every epoch. Normally, it is a good idea to shuffle,
    but for reproducibility, we have chosen to set the default to False.
    early_stopping is a boolean that indicates
    whether early stopping should be used
        - early stopping should only be performed if validation_data exists
        - early stopping should be based on validation loss
    patience is the patience used for early stopping

    Returns: the History object generated after training the model
    """

    def learning_rate_decay(epoch):
        """ callback that calculate the epic of the rate of a range"""
        alpha_utd = alpha / (1 + (decay_rate * epoch))
        return alpha_utd

    callbacks = []

    if early_stopping:
        EarlyStopping = Keras.callbacks.EarlyStopping(patience=patience)
        callbacks.append(EarlyStopping)

    if learning_rate_decay:
        decay = Keras.callbacks.LearningRateScheduler(
            learning_rate_decay,
            verbose=1
        )
        callbacks.append(decay)

    return network.fit(
        data,
        labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle,
        validation_data=validation_data,
        callbacks=callbacks
    )
