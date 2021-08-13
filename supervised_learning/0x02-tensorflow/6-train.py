#!/usr/bin/env python3
""" Module for neural network """
import tensorflow as tf
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations,
          alpha, iterations, save_path="/tmp/model.ckpt"):
    """
    that builds, trains, and saves a neural network classifier

    X_train is a numpy.ndarray containing the training input data
    Y_train is a numpy.ndarray containing the training labels
    X_valid is a numpy.ndarray containing the validation input data
    Y_valid is a numpy.ndarray containing the validation labels
    layer_sizes is a list containing the number of
    nodes in each layer of the network

    activations is a list containing the activation
    functions for each layer of the network

    alpha is the learning rate
    iterations is the number of iterations to train over
    save_path designates where to save the model

    Returns: the path where the model was saved
    """

    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])

    prediction = forward_prop(x, layer_sizes, activations)

    cost = calculate_loss(y, prediction)

    accuracy = calculate_accuracy(y, prediction)

    train = create_train_op(cost, alpha)

    saver = tf.train.Saver()

    session = tf.Session()
    session.run(
        tf.global_variables_initializer()
    )

    tf.add_to_collection(name="x", value=x)
    tf.add_to_collection(name="y", value=y)
    tf.add_to_collection(name="y_pred", value=prediction)
    tf.add_to_collection(name="loss", value=cost)
    tf.add_to_collection(name="accuracy", value=accuracy)
    tf.add_to_collection(name="train_op", value=train)

    for i in range(iterations + 1):
        session.run(prediction, feed_dict={x: X_train, y: Y_train})

        if i % 100 == 0 or i == iterations:
            print("After {} iterations:".format(i))
            print("\tTraining Cost: {}".format(
                session.run(cost, feed_dict={x: X_train, y: Y_train})
            )
            )

            print("\tTraining Accuracy: {}".format(
                session.run(accuracy, feed_dict={x: X_train, y: Y_train})
            )
            )

            print("\tValidation Cost: {}".format(
                session.run(cost, feed_dict={x: X_valid, y: Y_valid})
            )
            )

            print("\tValidation Accuracy: {}".format(
                session.run(accuracy, feed_dict={x: X_valid, y: Y_valid})
            )
            )

        if i < iterations:
            session.run(train, feed_dict={x: X_train, y: Y_train})

    return saver.save(session, save_path)
