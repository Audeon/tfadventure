import os
import re
import shutil
import string

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, losses
from functions import text_standardization_lc_punc

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

# Now we need to build a training, validation, and test dataset using the same method as the text classification,

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

raw_test_ds = tf.keras.utils.text_dataset_from_directory(
    test_dir,
    batch_size=batch_size
)

# Class names from raw_train_ds.classnames: ['csharp', 'java', 'javascript', 'python']

# Now we will do TextVectorization which will turn the words into ints when map them.
max_features = 10000
sequence_length = 250

vectorize_layer = layers.TextVectorization(
    standardize=text_standardization_lc_punc,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length
)

# Map the text only dataset (without labels) and adapt it.
# After calling adapt on a layer, a preprocessing layer's state will not update during training.
train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)

# Now to apply the vectorization to each dataset.
def text_vectorization(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label

train_ds = raw_train_ds.map(text_vectorization)
val_ds = raw_val_ds.map(text_vectorization)
test_ds = raw_test_ds.map(text_vectorization)

# Add some performance optimization.
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Now lets create the model
embedding_dim = 16
model = tf.keras.Sequential([
    layers.Embedding(max_features + 1, embedding_dim),
    layers.Dropout(0.2),
    layers.GlobalAveragePooling1D(),
    layers.Dropout(0.2),
    layers.Dense(4)
])

model.summary()

# Now lets add the loss function and optimization - This is a Multiclassification problem so we will use a
# SparseCategoricalCrossentorpy loss function.

model.compile(loss=losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer="adam",
              metrics=['accuracy'])

# Now train
epochs = 10
hisotry = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
)

loss, accuracy = model.evaluate(test_ds)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

# Plot Stuff
history_dict = hisotry.history
history_dict.keys()

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc)+1)

plt.plot(epochs, loss, 'bo', label="Training Loss")
plt.plot(epochs, val_loss, 'b', label="Validation Loss")

plt.title("Training and Validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.show()

plt.plot(epochs, acc, 'bo', label="Training Accuracy")
plt.plot(epochs, val_acc, 'b', label="Validation Accuracy")

plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.show()