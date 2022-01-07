#!/usr/bin/env python3
""" Transformer Applications """
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset():
    """ Dataset loads and preps a dataset for machine translation """

    def __init__(self):
        """ Constructor """
        self.data_train = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='train',
            as_supervised=True
        )
        self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='validation',
            as_supervised=True
        )
        tokenizer_pt, tokenizer_en = self.tokenize_dataset(self.data_train)
        self.tokenizer_pt = tokenizer_pt
        self.tokenizer_en = tokenizer_en

    def tokenize_dataset(self, data):
        """
        creates sub-word tokenizers for our dataset

        Args:
            data (tuple): (pt, en)
                pt is the tf.Tensor containing the Portuguese sentence
                en is the tf.Tensor containing the corresponding English
                sentence

        Returns:
            tokenizer_pt is the Portuguese tokenizer
            tokenizer_en is the English tokenizer
        """
        subword = tfds.features.text.SubwordTextensorflow_texttEncoder.build_from_corpus
        # subword = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus
        tokenizer_en = subword(
            (en.numpy() for pt, en in data),
            target_vocab_size=2**15
        )

        tokenizer_pt = subword(
            (pt.numpy() for pt, en in data),
            target_vocab_size=2**15
        )
        return (tokenizer_pt, tokenizer_en)

    def encode(self, pt, en):
        """
        encodes a translation into tokens
            The tokenized sentences should include the start and
            end of sentence tokens

        Args:
            pt (tf.Tensor):  the Portuguese sentence
            en (tf.Tensor): the corresponding English sentence

        Returns:
            tokenizer_pt is the Portuguese tokenizer
            tokenizer_en is the English tokenizer
        """

        pt = \
            [self.tokenizer_pt.vocab_size] + \
            self.tokenizer_pt.encode(pt.numpy()) + \
            [self.tokenizer_pt.vocab_size + 1]

        en = \
            [self.tokenizer_en.vocab_size] + \
            self.tokenizer_en.encode(en.numpy()) + \
            [self.tokenizer_en.vocab_size + 1]

        return pt, en
