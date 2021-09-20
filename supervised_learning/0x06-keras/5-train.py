#!/usr/bin/env python3
""" Train the model using keras """


import tensorflow.keras as Keras


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    """
    Training the model cnn with the validation data

    Args:
        network ([type]): [description]
        data ([type]): [description]
        labels ([type]): [description]
        batch_size ([type]): [description]
        epochs ([type]): [description]
        validation_data ([type], optional): [description]. Defaults to None.
        verbose (bool, optional): [description]. Defaults to True.
        shuffle (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    return network.fit(
        data,
        labels,
        batch_size=batch_size,
        epochs=epochs,
        shuffle=shuffle,
        verbose=verbose,
        validation_data=validation_data
    )
