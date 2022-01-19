#!/usr/bin/env python3
""" QA Bot """

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
from transformers import BertTokenizer


SEMANTIC_MODEL = hub.load(
    "https://tfhub.dev/google/universal-sentence-encoder-large/5"
)
TZ = BertTokenizer.from_pretrained(
    'bert-large-uncased-whole-word-masking-finetuned-squad'
)
MODEL = hub.load('https://tfhub.dev/see--/bert-uncased-tf2-qa/1')
EXIT_COMMANDS = ['exit', 'quit', 'goodbye', 'bye']


def format_question(answ):
    """
    Getting the answear if the answear not match with the message
    throw a deafult message for formatting the text

    Args:
        answ (string): text that contain the answear

    Return:
        the proper answear
    """

    if answ is None or answ is "":
        answ = "Sorry, I do not understand your question."

    return answ


# loop to get the answers
def question_answer(coprus_path):
    """
    answers questions from multiple reference texts

    Args:
        coprus_path (): the path to the corpus of reference
    """
    try:
        while True:
            user_response = input("Q: ")
            if user_response.lower() in EXIT_COMMANDS:
                print("A: Goodbye")
                break
            answer = format_question(q_answer(
                # formatting and validate not null values
                # get the answer by semantic search
                user_response,
                semantic_search(coprus_path, user_response)
            ))
            print("A: {}".format(answer))

    except KeyboardInterrupt:
        print("\nA: Goodbye")
        exit(0)


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
    documents = [sentence]
    for filename in os.listdir(corpus_path):
        if filename.endswith(".md") is False:
            continue

        with open(corpus_path + "/" + filename, "r", encoding="utf-8") as f:
            documents.append(f.read())

    embed = SEMANTIC_MODEL(documents)
    close = np.argmax(np.inner(embed, embed)[0, 1:])  # correlation line

    return documents[close + 1]  # most approx


def q_answer(question, reference):
    """
    finds a snippet of text within a document to answer question

    Args:
        question (str): question to answer
        reference (str):  a string containing the reference
            document from which to find the answer

    returns:
        The proper answer to the question
    """
    question_tkz = TZ.tokenize(question)
    paragraph_tks = TZ.tokenize(reference)
    tokens = ['[CLS]'] + question_tkz + ['[SEP]'] + paragraph_tks + ['[SEP]']
    word_ids = TZ.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(word_ids)
    type_ids = [0] * (1 + len(question_tkz) + 1) + \
        [1] * (len(paragraph_tks) + 1)

    word_ids, input_mask, type_ids = map(
        lambda t: tf.expand_dims(
            tf.convert_to_tensor(t, dtype=tf.int32), 0
        ), (word_ids, input_mask, type_ids)
    )

    outputs = MODEL([word_ids, input_mask, type_ids])

    short_start = tf.argmax(outputs[0][0][1:]) + 1  # enforce an answer
    short_end = tf.argmax(outputs[1][0][1:]) + 1

    answer = TZ.convert_tokens_to_string(
        tokens[short_start: short_end + 1]  # answer is inclusive
    )

    if answer:
        return answer
    # If no answer is found, return None
    return None
