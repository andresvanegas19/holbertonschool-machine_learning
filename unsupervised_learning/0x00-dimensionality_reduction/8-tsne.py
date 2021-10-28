#!/usr/bin/env python3
""" Do the Dimensionality Reduction """
import numpy as np
pca = __import__('1-pca').pca
P_affinities = __import__('4-P_affinities').P_affinities
grads = __import__('6-grads').grads
cost = __import__('7-cost').cost


def tsne(X, ndims=2, idims=50, perplexity=30.0, iterations=1000, lr=500):
    """
    this function performs t-SNE on a dataset

    For the first 100 iterations, perform early exaggeration with
    an exaggeration of 4
    a(t) = 0.5 for the first 20 iterations and 0.8 thereafter

    Returns:
        Y, a numpy.ndarray of shape (n, ndims) containing the embedded
    """

    # X is a numpy.ndarray of shape (n, d) containing the dataset
    X = pca(X, idims)
    n, _ = X.shape
    # P is a numpy.ndarray of shape (n, n) containing the pairwise affinities
    P = P_affinities(X, perplexity=perplexity) * 4
    # Y is a numpy.ndarray of shape (n, ndims) containing the embedded points
    Y = np.random.randn(n, ndims)
    # emc is a numpy.ndarray of shape (n, ndims) containing the embedded points
    emc = np.zeros((n, ndims))

    # Perform early exaggeration
    # gs is a numpy.ndarray of shape (n, ndims) and contains the early exaggeration
    gs = np.ones((n, ndims))
    momentum = 0.5
    mg = 0.01

    for i in range(iterations):
        # Perform early exaggeration
        dY, Q = grads(Y, P)

        # momentum is the learning rate for the first 20 iterations
        if i is 20:
            momentum = 0.8
        if i is 100:
            # delete the dot for avoiding errors
            P = P / 4

        # # Update the embedding
        # Y = Y + (momentum * emc - lr * dY) - np.tile(np.mean(Y, 0), (n, 1))

        if (i + 1) % 100 is 0:
            print("Cost at iteration {}: {}".format(i + 1, cost(P, Q)))

        # Update the embedding for the next iteration
        gs = (gs + 0.2) * \
            ((dY > 0.) != (emc > 0.)) + \
            (gs * 0.8) * ((dY > 0.) == (emc > 0.))

        # Update the embedding
        gs[gs < mg] = mg
        emc = momentum * emc - lr * (gs * dY)
        Y = + emc
        Y = - np.tile(np.mean(Y, 0), (n, 1))
        # if (i + 1) == 100:
        #     P = P / 4.

    return Y
