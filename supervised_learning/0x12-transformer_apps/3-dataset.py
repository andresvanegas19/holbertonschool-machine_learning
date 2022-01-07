#!/usr/bin/env python3
""" Transformer Applications """
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset():
    """ Dataset loads and preps a dataset for machine translation """

    def __init__(self, batch_size, max_len):
        """ Constructor """
        examples, metadata = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            with_info=True,
            as_supervised=True
        )

        self.data_train = examples['train']
        self.data_valid = examples['validation']

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train  # sub-word tokenizers
        )
        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

        def max_len_tf(x, y, max_len=max_len):
            """
            filter max length of the sequence

            Args:
                x: the Portuguese sentence
                y: the corresponding English sentence
                max_len: the maximum length of the sequence

            Returns:
                returns the filtered sequence
            """
            return tf.logical_and(
                tf.size(x) <= max_len,
                tf.size(y) <= max_len
            )

        # max_len_tf = lambda x, y: tf.logical_and(
        self.data_train = self.data_train.filter(max_len_tf)
        self.data_train = self.data_train.cache()

        buffer_size = metadata.splits['train'].num_examples
        # buffer_size = 10000
        self.data_train = self.data_train.shuffle(buffer_size).padded_batch(
            batch_size, padded_shapes=([None], [None])
        )
        self.data_train = self.data_train.prefetch(
            tf.data.experimental.AUTOTUNE  # .sample_from_dataset
        )

        self.data_valid = self.data_valid.filter(max_len_tf)
        self.data_valid = self.data_valid.padded_batch(
            batch_size, padded_shapes=([None], [None])
        )

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
        # subword = tfds.features.text.SubwordTextensorflow_texttEncoder.build_from_corpus
        subword = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus
        tokenizer_en = subword(
            (en.numpy() for _, en in data),
            target_vocab_size=2**15
        )

        tokenizer_pt = subword(
            (pt.numpy() for pt, _ in data),
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

    def tf_encode(self, pt, en):
        """
        acts as a tensorflow wrapper for the encode instance method

        Args:
            pt (tf.Tensor):  the Portuguese sentence
            en (tf.Tensor): the corresponding English sentence
        """
        result_port, result_engl = tf.py_function(
            self.encode,
            [pt, en],
            [tf.int64, tf.int64]
        )

        result_port.set_shape([None])
        result_engl.set_shape([None])

        return result_port, result_engl
