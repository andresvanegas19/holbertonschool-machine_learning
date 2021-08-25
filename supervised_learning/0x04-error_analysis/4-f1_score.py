#!/usr/bin/env python3
""" Module for Error Analysis method and technicths"""
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    calculates the F1 score of a confusion matrix:

    confusion is a confusion numpy.ndarray of shape (classes, classes)
    where row indices represent the correct labels and column indices
    represent the predicted labels
        - classes is the number of classes

    Returns: a numpy.ndarray of shape (classes,)
    containing the F1 score of each class
    """

    ppv = precision(confusion)
    tpr = sensitivity(confusion)

    return 2 * ppv * tpr / (ppv + tpr)
