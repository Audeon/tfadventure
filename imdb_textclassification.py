#!/usr/bin/env python
import os
import re
import shutil
import string

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
test_dir = os.path.join(dataset_dir, 'test')
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
    subset='training', # <------------Notice here this determines which set the data belongs to, this tells it to leave 20% of the datafiles in the directy for the next call of this.
    seed=seed
)

# Now that we have it stored in tf.data we can iterate through it.
for text_batch, label_batch in raw_train_ds.take(1):
    for i in range(3):
        print("Review: ", text_batch.numpy()[i])
        print("Label: ", label_batch.numpy()[i])

print("Label 0 corresponds to", raw_train_ds.class_names[0])
print("Label 1 corresponds to", raw_train_ds.class_names[1])

# Now we are going to take the remaining documents and create the validation dataset. And the test dataset.
raw_val_ds = tf.keras.utils.text_dataset_from_directory(
    train_dir,
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation', # <------------Notice here this determines which set the data belongs to. because of this it knows to take the last 20% of data files from the directy
    seed=seed )

raw_test_ds = tf.keras.utils.text_dataset_from_directory(
    test_dir,
    batch_size=batch_size )

# TODO: This would generally be in an imported function when we start breaking up the code in a proper problem.
# Now we build a function to strip the HTML and punctuation from the input data.
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html, '[%s]' % re.escape(string.punctuation), '')

# Now we are goign to create the TextVectorization layer which will standardize, tokenize, and vectorize the data for the NN.
max_features = 10000
sequence_lenght = 250

vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_lenght )

# Make a text-only dataset (without labels), then call adapt.
train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)

# Now lets visualize some data.
def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label

# Grab a batch of 32 reviews and labels from the dataset
view_tb, view_lb = next(iter(raw_train_ds))
first_review, first_label = view_tb[0], view_lb[0]
print("Review: ", first_review)
print("Label", raw_train_ds.class_names[first_label])
print("Vectorized review", vectorize_text(first_review, first_label))

# We can look at what the vocabularay is for each of the above vectorized ints.
print("85 ---> ",vectorize_layer.get_vocabulary()[85])
print("17 ---> ",vectorize_layer.get_vocabulary()[17])
print("260 ---> ",vectorize_layer.get_vocabulary()[260])
print("2 ---> ",vectorize_layer.get_vocabulary()[2])
print('Vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))

# Now lets apply the same textvectorization layer we created
train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

# Now configure the datasets for performance. We can cache them in memory to ensure loading the dataset is not the
# bottle neck of training it. We will use prefetch to overlap data processing and model execution while training.
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Now to create the model for the Neural Network
embedding_dim = 16
model = tf.keras.Sequential([
    layers.Embedding(max_features + 1, embedding_dim),
    layers.Dropout(0.2),
    layers.GlobalAveragePooling1D(),
    layers.Dropout(0.2),
    layers.Dense(1)])

model.summary()

# Loss function and optimization - This is a binary classification problem so we will need a loss function and optimizer.
model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

# Now we train the model
call_back = tf.keras.callbacks.EarlyStopping(monitor='val_binary_accuracy', patience=2, min_delta=0.02)
epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=call_back
)

# Eval the model

loss, accuracy = model.evaluate(test_ds)
print("Loss: ", loss)
print("Accuracy: ", accuracy)


# Model.fit returns a history that we can plot over time.
history_dict = history.history
history_dict.keys()

acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc)+1)
# "bo" is for blue dot.
plt.plot(epochs, loss, 'bo', label="Training Loss")

# 'b' is for solid blue line
plt.plot(epochs, val_loss, 'b', label="Validation Loss")

plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.show()

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.show()

# WE will include the text vectorization layer inside the model in order to simplyfy processing raw strings
# This will make deployment and exporting easier.

export_model = tf.keras.Sequential([
    vectorize_layer,
    model,
    layers.Activation('sigmoid')
])

export_model.compile(
    loss=losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
)

# Now test it with raw_test_ds, which yields raw strings.

loss, accuracy = export_model.evaluate(raw_test_ds)
print(accuracy)