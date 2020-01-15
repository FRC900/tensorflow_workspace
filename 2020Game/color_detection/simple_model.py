from tensorflow.keras import layers, models, optimizers, losses, regularizers
from keras.callbacks import ModelCheckpoint
import numpy as np
import tensorflow as tf
import datetime

model = models.Sequential([
    layers.Dense(20, input_dim=6, kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)),
    layers.Activation("relu"),
    layers.Dropout(0.2),
    layers.Dense(4, kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)),
    layers.Activation("softmax")
])

model.summary()

model.compile("adam", "categorical_crossentropy", metrics=["categorical_accuracy"])

data = np.load("data.npz", allow_pickle=True)

batch_size = 64

reorder_data = np.array(list(range(0,len(data['X']))))
np.random.shuffle(reorder_data)

X = data['X'][reorder_data]
y = data['Y'][reorder_data]

X_train = X[:800]
y_train = y[:800]

X_validation = X[800:]
y_validation = y[800:]

max_acc = 0
model.fit(X_train, y_train, epochs=30, validation_data=(X_validation, y_validation), batch_size=batch_size, verbose=1)

y_pred = np.argmax(model.predict(X_validation), axis=-1)
y = np.argmax(y_validation, -1)
print(sum((y_pred-y)**2))