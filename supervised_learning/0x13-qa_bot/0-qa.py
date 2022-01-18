#!/usr/bin/env python3
""" QA Bot """

import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(question, reference):
    """
    finds a snippet of text within a document to answer question

    Args:
        question (str): question to answer
        reference (str):  a string containing the reference
            document from which to find the answer

    returns:
        The proper answer to the question
    """

    tz = BertTokenizer.from_pretrained(
        'bert-large-uncased-whole-word-masking-finetuned-squad'
    )
    model = hub.load('https://tfhub.dev/see--/bert-uncased-tf2-qa/1')

    question_tkz = tz.tokenize(question)
    paragraph_tks = tz.tokenize(reference)
    tokens = ['[CLS]'] + question_tkz + ['[SEP]'] + paragraph_tks + ['[SEP]']
    word_ids = tz.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(word_ids)
    type_ids = [0] * (1 + len(question_tkz) + 1) + \
        [1] * (len(paragraph_tks) + 1)

    word_ids, input_mask, type_ids = map(
        lambda t: tf.expand_dims(
            tf.convert_to_tensor(t, dtype=tf.int32), 0
        ), (word_ids, input_mask, type_ids)
    )

    outputs = model([word_ids, input_mask, type_ids])

    short_start = tf.argmax(outputs[0][0][1:]) + 1  # enforce an answer
    short_end = tf.argmax(outputs[1][0][1:]) + 1

    answer = tz.convert_tokens_to_string(
        tokens[short_start: short_end + 1]  # answer is inclusive
    )

    return answer
