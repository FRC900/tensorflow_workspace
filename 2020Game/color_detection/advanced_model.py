from keras import layers, models, optimizers, losses, regularizers
from keras.callbacks import ModelCheckpoint, TensorBoard
import numpy as np

encoder = models.Sequential([
    layers.Dense(64, input_dim=6, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(4, activation="sigmoid")
])

decoder = models.Sequential([
    layers.Dense(36, input_dim=4),
    layers.LeakyReLU(0.4),
    layers.Dense(16),
    layers.LeakyReLU(0.4),
    layers.Dense(16),
    layers.LeakyReLU(0.4),
    layers.Dense(16),
    layers.LeakyReLU(0.4),
    layers.Dense(16),
    layers.LeakyReLU(0.4),
    layers.Dense(16),
    layers.LeakyReLU(0.4),
    layers.Dense(16),
    layers.LeakyReLU(0.4),
    layers.Dense(6)
])

encoder.compile(optimizers.Nadam(), "categorical_crossentropy", metrics=['categorical_accuracy'])

inp = layers.Input(shape=(6,))
total_model = models.Model(inputs=inp, outputs=decoder(encoder(inp)))
total_model.compile(optimizers.Nadam(), "mse")

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

X_train = X[:8000]
y_train = y[:8000]

X_validation = X[8000:]
y_validation = y[8000:]

# total_model.fit(X_train, X_train, validation_data=(X_validation, X_validation), epochs=5, verbose=1)
for epoch in range(100):
    print("="*10, epoch, "="*10)
    encoder.fit(X_train, y_train, validation_data=(X_validation, y_validation), epochs=10, verbose=1)
    encoder.save("detect_adv.h5")