#!/usr/bin/env python3
""" Module for neural network """
import tensorflow as tf


def evaluate(X, Y, save_path):
    """
    that evaluates the output of a neural network:

    X is a numpy.ndarray containing the input data to evaluate
    Y is a numpy.ndarray containing the one-hot labels for X
    save_path is the location to load the model from

    Returns: the network’s prediction, accuracy, and loss, respectively
    """
    with tf.Session() as session:

        tf.train.import_meta_graph(save_path + ".meta") \
            .restore(session, save_path)

        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]

        y_prediction = tf.get_collection("y_pred")[0]

        prediction = session.run(y_prediction, feed_dict={x: X, y: Y})

        accuracy = session.run(
            tf.get_collection("accuracy")[0],
            feed_dict={x: X, y: Y}
        )

        loss = tf.get_collection("loss")[0]
        cost = session.run(loss, feed_dict={x: X, y: Y})

    return prediction, accuracy, cost
