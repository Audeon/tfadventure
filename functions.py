#!/usr/bin/env python
import re
import string

import tensorflow as tf


def text_standardization_lc_punc(input_data):
    lowercase = tf.strings.lower(input_data)
    return tf.strings.regex_replace(lowercase, '[%s]' % re.escape(string.punctuation), '')

def text_vectorization(text, label, vectorize_layer):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label