#!/usr/bin/env python3
""" Natural Language Processing - Word Embeddings """
import numpy as np
from keras.layers import Embedding


def gensim_to_keras(model):
    """
    converts a gensim word2vec model to a keras Embedding layer

    Args:
        model is a trained gensim word2vec models

    Returns:
        the trainable keras Embedding
    """
    vocab = model.wv.vocab
    emb_matric = np.zeros((len(vocab), model.vector_size))

    for i in range(len(vocab)):
        embedding_vector = model.wv[model.wv.index2word[i]]
        if embedding_vector is not None:
            emb_matric[i] = embedding_vector

    return Embedding(
        input_dim=emb_matric.shape[0],
        output_dim=emb_matric.shape[1],
        weights=[emb_matric],
    )
