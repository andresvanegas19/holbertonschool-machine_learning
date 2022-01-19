#!/usr/bin/env python3
""" QA Bot """
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


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


def answer_loop(reference):
    """
    A loop that wait the typing of the user

    Args:
        reference: is the reference text
    """
    try:
        while True:
            response = input("Q: ")
            if response.lower() in EXIT_COMMANDS:
                print("A: Goodbye")
                break
            answer = format_question(
                question_answer(response, reference)
            )

            print("A: {}".format(answer))
    except KeyboardInterrupt:
        print("\nA: Goodbye")
        exit(0)


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
