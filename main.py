import os
import re
import shutil
import string

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, losses

print(tf.__version__)


# First download the data into the directory we want it under ./data
base_dir = os.path.dirname(os.path.realpath(__file__))
url = "https://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar.gz"

cache_dir = os.path.join(base_dir, "data")
data_path = os.path.join(cache_dir, "stack_overflow_16k")
if not os.path.exists(data_path):
    try:
        os.mkdir(data_path)
    except Exception as err:
        print(err)

dataset = tf.keras.utils.get_file("stack_overflow_16k", url, untar=True, cache_dir=data_path, cache_subdir='')
train_dir = os.path.join(data_path, "train")
test_dir = os.path.join(data_path, "test")

batch_size = 32
seed = 56

# Now we need to build a training and validation set using the same method as the text classification,

raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    train_dir,
    batch_size=batch_size,
    validation_split = 0.25,
    subset="training",
    seed=seed
)

raw_val_ds = tf.keras.utils.text_dataset_from_directory(
    train_dir,
    batch_size=batch_size,
    validation_split= 0.25,
    subset="validation",
    seed=seed
)

for text, label in raw_train_ds.take(1):
    for i in range(10):
        print(f"Question: {text.numpy()[i]}")
        print(f"Classification: {raw_train_ds.class_names[label.numpy()[i]]}")
