#!/usr/bin/env python3
""" Natural Language Processing Metrics """
import numpy as np


def uni_bleu(references, sentence):
    """
    calculates the unigram BLEU score for a sentence:

    Args:
        references: is a list of reference translations
            - each reference translation is a list
            of the words in the translation
        sentence: is a list containing the model proposed sentence

    Returns:
        the unigram BLEU score
    """
    sen_len = len(sentence)
    ref_len = []
    words = {}

    for transl in references:
        ref_len.append(len(transl))
        for word in transl:
            if word in sentence and word not in words.keys():
                words[word] = 1

    # print(words)
    index = np.argmin([abs(len(i) - sen_len) for i in references])
    # print(index)
    best_match = len(references[index])

    # this conditional is to avoid division by zero
    if sen_len > best_match:
        BLEU = 1
    else:
        BLEU = np.exp(1 - float(best_match) / float(sen_len))

    # BLEU score
    return BLEU * np.exp(np.log(sum(words.values()) / sen_len))
