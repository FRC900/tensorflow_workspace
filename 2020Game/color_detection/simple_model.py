from tensorflow.keras import layers, models, optimizers, losses, regularizers
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np

# Simple fully-connected net with 1 hidden layer
# Input is the 6 color channel values from the color sensor
# Output is 4 values, each with a softmax value for 1 of the 4 color wheel colors
model = models.Sequential([
    layers.Dense(12, input_dim=6, kernel_regularizer=regularizers.l1_l2(l1=0.005, l2=0.005)),
    layers.Activation("relu"),
    layers.Dropout(0.2),
    layers.Dense(4, kernel_regularizer=regularizers.l1_l2(l1=0.005, l2=0.005)),
    layers.Activation("softmax")
])

model.summary()

model.compile("adam", "categorical_crossentropy", metrics=["categorical_accuracy"])

data = np.load("unnormalized_data.npz", allow_pickle=True)

batch_size = 64
# 30-40 epochs get it to the flat part of the curve, but it continues improving marginally beyond that...
num_epochs = 700

# Shuffle X&Y data into random order
reorder_data = np.array(list(range(0,len(data['X']))))
np.random.shuffle(reorder_data)

X = data['X'][reorder_data]
Y = data['Y'][reorder_data]

# Do an 85/15 training / validation split of the data
train_data_len = int(len(data['X']) * .85)
X_train = X[:train_data_len]
Y_train = Y[:train_data_len]

X_validation = X[train_data_len:]
Y_validation = Y[train_data_len:]

model.fit(X_train, Y_train, epochs=num_epochs, validation_data=(X_validation, Y_validation), batch_size=batch_size, verbose=1)

y_pred = np.argmax(model.predict(X_validation), axis=-1)
y = np.argmax(Y_validation, -1)
print(sum((y_pred-y)**2))

model.save('unnormalized_data.model', save_format='tf')

