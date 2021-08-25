#!/usr/bin/env python3
""" Module for Error Analysis method and technicths"""
import numpy as np


def specificity(confusion):
    """
    calculates the specificity for each class in a confusion matrix:

    confusion is a confusion numpy.ndarray of shape (classes, classes
    where row indices represent the correct labels and column indices
    represent the predicted labels
        - classes is the number of classes

    Returns: a numpy.ndarray of shape (classes,) containing
    the specificity of each class
    """
    # When there are more than two classes in a confusion matrix,
    # specificity is not a useful metric as there are inherently
    # more actual negatives than actual positives. It is much better
    # o use sensitivity (recall) and precision.

    fp = confusion.sum(axis=0) - np.diagonal(confusion)

    tn = confusion.sum() - \
        (fp + (
            confusion.sum(axis=1) -
            np.diagonal(confusion)
        ) + np.diagonal(confusion))

    return tn / (tn + fp)
