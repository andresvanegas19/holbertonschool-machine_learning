#!/usr/bin/env python3
""" Natural Language Processing - Word Embeddings """
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """
    creates a TF-IDF embedding

    Args:
        sentences is a list of sentences to analyze
        vocab(list): to use for the analysis
            - If None, all words within sentences should be used

    Returns:
        embeddings is a numpy.ndarray of shape (s, f) containing the embeddings
            - s(number): sentences in sentences
            - f(number): features analyzed
        features is a list of the features used for embeddings
    """
    TfidfV_instance = TfidfVectorizer(vocabulary=vocab)
    embeddings = TfidfV_instance.fit_transform(sentences)
    features = TfidfV_instance.get_feature_names()

    return embeddings.toarray(), features
