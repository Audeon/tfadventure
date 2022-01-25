import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

clas_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Exploring the format of the data
print(f"train_images.shape: {train_images.shape}")
print(f"len(train_labels): {len(train_labels)}")

print(f"Each train label is between 0 - 9: {train_labels}")


# Lets preproccess the data

# Graph time!
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# We are scaling the values of these images to a range of 0 to 1.
train_images = train_images / 255.0
test_images = test_images / 255.0

# Now verify the data is correct and that your ready to build a NN - Display the first 25 images.

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5, i+1)
    plt.xticks([])
    plt.yticks()
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(clas_names[train_labels[i]])

plt.show()

# Lets build the layers of the Nerual network

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

# Compiling the model and adding a loss function optimizer and metrics
model.compile(optimizer='adam',
             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
             metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)