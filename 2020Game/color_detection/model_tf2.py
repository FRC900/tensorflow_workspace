from tensorflow.keras import layers, models, optimizers, losses, regularizers
import numpy as np

model = models.Sequential([
    layers.Dense(20, input_dim=6, kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)),
    layers.Activation("relu"),
    layers.Dropout(0.2),
    layers.Dense(4, kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)),
    layers.Activation("softmax")
])

model.summary()

model.compile("adam", "binary_crossentropy", metrics=["accuracy"])

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

for i in range(10000):
    print("="*10, "EPOCH", i, "="*10)
    model.fit(X_train, y_train, epochs=100, batch_size=batch_size, verbose=0)
    acc = model.evaluate(X_validation, y_validation)
    print(acc[1])
    if(float(acc[1]) > max_acc):
        max_acc = float(acc[1])
        model.save("color_detector.h5")
