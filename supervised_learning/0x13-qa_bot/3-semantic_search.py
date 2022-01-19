#!/usr/bin/env python3
""" QA Bot """

import tensorflow_hub as hub
import os
import numpy as np


def semantic_search(corpus_path, sentence):
    """
    performs semantic search on a corpus of documents

    Args:
        corpus_path (): the path to the corpus of reference
            documents on which to perform semantic search
        sentence (): the sentence from which to perform semantic search

    Returns:
        the reference text of the document most similar to sentence
    """
    articles = [sentence]

    for filename in os.listdir(corpus_path):
        if not filename.endswith('.md'):
            continue

        with open(corpus_path + '/' + filename, 'r', encoding='utf-8') as file:
            articles.append(file.read())

    embed = hub.load(
        "https://tfhub.dev/google/universal-sentence-encoder-large/5"
    )

    embeddings = embed(articles)
    closest = np.argmax(np.inner(embeddings, embeddings)[0, 1:])
    print(closest)

    return articles[closest + 1]
