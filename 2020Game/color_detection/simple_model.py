import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, regularizers
#from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    https://stackoverflow.com/questions/45466020/how-to-export-keras-h5-to-tensorflow-pb
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph

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

# Save the model, first as a keras model, then convert
# that keras model to a TF lite saved file
from tensorflow.keras import backend as K
model.save('unnormalized_data.h5')
#frozen_graph = freeze_session(K.get_session(),
#                             output_names=[out.op.name for out in model.outputs])

#tf.train.write_graph(frozen_graph, ".", "simple_model.pb", as_text=False)
converter = tf.lite.TFLiteConverter.from_keras_model_file('unnormalized_data.h5')
tflite_model = converter.convert()
open("unnormalized_data.tflite", "wb").write(tflite_model)

# Also optimize the model for performance, not sure if this 
# really will make a difference with something so small
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_fp16_model = converter.convert()
open("unnormalized_data_optimized.tflite", "wb").write(tflite_fp16_model)

trainable_count = int(
    np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
non_trainable_count = int(
    np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))

print('Total params: {:,}'.format(trainable_count + non_trainable_count))
print('Trainable params: {:,}'.format(trainable_count))
print('Non-trainable params: {:,}'.format(non_trainable_count))
