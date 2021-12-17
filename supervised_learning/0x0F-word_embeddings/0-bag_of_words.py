#!/usr/bin/env python3
""" Natural Language Processing - Word Embeddings """
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """
    creates a bag of words embedding matrix

    Args:
        sentences: is a list of sentences to analyze
        vocab: is a list of the vocabulary words to use for the analysis
            - If None, all words within sentences should be used

    Returns:
        embeddings is a numpy.ndarray of shape (s, f) containing the embeddings
            - s(number): sentences in sentences
            - f(number): features analyzed
        features is a list of the features used for embeddings
    """
    cv = CountVectorizer(vocabulary=vocab)
    embeddings = cv.fit_transform(sentences)
    # Obtain the features
    features = cv.get_feature_names()
    emb = embeddings.toarray()

    return emb, features
