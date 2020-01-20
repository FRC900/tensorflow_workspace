from tensorflow.keras import layers, models, optimizers, losses, regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import numpy as np
import tensorflow as tf
import datetime

l1 = 0
l2 = 0

model = models.Sequential([
    layers.Dense(50, input_dim=6, kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)),
    layers.Activation("relu"),
    layers.Dropout(0.5),
    layers.Dense(50, kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)),
    layers.Activation("relu"),
    layers.Dropout(0.5),
    layers.Dense(50, kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)),
    layers.Activation("relu"),
    layers.Dropout(0.5),
    layers.Dense(50, kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)),
    layers.Activation("relu"),
    layers.Dropout(0.5),
    layers.Dense(4, kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)),
    layers.Activation("softmax")
])

model.summary()

lr=1e-3

model.compile(optimizers.Nadam(lr=lr), "categorical_crossentropy", metrics=["categorical_accuracy"], callbacks=[
    ModelCheckpoint("model.{epochs:02d}.{val_acc:04f}.h5", "val_acc"), TensorBoard()])

data_angle = np.load("data_angle.npz", allow_pickle=True)
data_straight = np.load("data_straight.npz", allow_pickle=True)
data = np.load("data.npz", allow_pickle=True)

batch_size = 10

reorder_data = np.array(list(range(0,len(data['X'])+len(data_angle['X'])+len(data_straight['X']))))
np.random.shuffle(reorder_data)

# print([i.shape for i in (data['Y'], data_straight['Y'], data_angle['Y'])])

X = np.concatenate((data['X'], data_straight['X'], data_angle['X']), axis=0)[reorder_data]
y = np.concatenate((data['Y'], data_straight['Y'], data_angle['Y']), axis=0)[reorder_data]

print(X.shape)
print(y.shape)

X_train = X[:5000]
y_train = y[:5000]

X_validation = X[5000:]
y_validation = y[5000:]

print(1/len(X_validation))

max_acc = 0
model.fit(X_train, y_train, epochs=30, validation_data=(X_validation, y_validation), batch_size=batch_size, verbose=1)
model.save("detect.h5")

y_pred = np.argmax(model.predict(X), axis=-1)
y = np.argmax(y, -1)
print(sum((y_pred-y)**2))