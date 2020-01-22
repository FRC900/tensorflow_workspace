import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(500),
    layers.LeakyReLU(0.4),
    layers.Dense(300),
    layers.LeakyReLU(0.4),
    layers.Dense(10),
    layers.Activation("softmax")
])

model.summary()

model.compile(optimizers.Nadam(), losses.SparseCategoricalCrossentropy(), metrics=['sparse_categorical_crossentropy'])
model.fit(x_train, y_train, validation_split=0.2)
