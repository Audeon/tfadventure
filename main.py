import os
import re
import shutil
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, losses

print(tf.__version__)


url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

cache_dir = "./data"

if not os.listdir(cache_dir):
    dataset = tf.keras.utils.get_file("aclImdb_v1", url, untar=True, cache_dir=cache_dir, cache_subdir='')

dataset_dir = os.path.join(cache_dir, 'aclImdb')
train_dir = os.path.join(dataset_dir, "train")

# Lets read one for fun
sample_file = os.path.join(train_dir, 'pos/11_9.txt')
with open(sample_file) as f:
    print(f.read())

# Now we need to remove the data directories we are not interested in, in this case the unsup reviews.
remove_dir = os.path.join(train_dir, 'unsup')
if os.path.lexists(remove_dir):
    shutil.rmtree(remove_dir)

# Next we are going to split up the training dataset in order to have a training set and a validation set as we
# arleady have atest set from this data

batch_size = 32
seed = 42

raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    train_dir,
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=seed
)
