#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def input_fn(features,labels, training=True, batch_size=256):
    # Convert inputs to dataset
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    if training:
        dataset = dataset.shuffle(1000).repeat()

    return dataset.batch(batch_size)


if __name__ == '__main__':

    csv_column_names = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
    species = ['Setosa', 'Versicolor', 'Virginica']


    train = pd.read_csv("./iris_training.csv", names=csv_column_names, header=0)
    test = pd.read_csv("./iris_test.csv", names=csv_column_names, header=0)

    train_y = train.pop('Species')
    test_y = test.pop('Species')

    feature_columns = []
    for key in train.keys():
        feature_columns.append(tf.feature_column.numeric_column(key=key))

    # Next we build the Deep Neural Network (DNN)
    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        # Two hidden layers of 30 and 10 nodes respectively
        hidden_units=[30, 10],
        # The model must choose between 3 classes.
        n_classes=3
    )

    # WE use lamdba to avoid creating an inner function with our input function.
    classifier.train(
        input_fn=lambda: input_fn(train, train_y, training=True), steps=5000
    )

    eval_result = classifier.evaluate(
        input_fn=lambda: input_fn(test, test_y, training=False))

    print(f"Test set accuracy: { eval_result}")