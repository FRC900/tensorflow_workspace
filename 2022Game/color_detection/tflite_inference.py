import tflite_runtime.interpreter as tflite
import numpy as np

# Create interpreter from saved model file - see simple_model.py 
# for how to create and train this model
tflite_model_file = 'unnormalized_data.tflite'
interpreter = tflite.Interpreter(model_path=str(tflite_model_file))
interpreter.allocate_tensors()

# Might get more speed using 16-bit floats rather than 32-bit,
# but not sure if it'll really matter that much
#tflite_model_fp16_file = 'unnormalized_data_optimized.tflite'
#interpreter_fp16 = tflite.Interpreter(model_path=str(tflite_model_fp16_file))
#interpreter_fp16.allocate_tensors()

# Input shape of the model
input_details = interpreter.get_input_details()[0]
input_shape = input_details['shape']
print (input_shape)

# Example test input - np array of arrays of shape input.
# Think the outer array allows batching inputs, that is,
# each index of the outer array is a set of data to 
# run inference on at once
test_input = np.array([[455,246,388,566,1288,996]]).astype(np.float32)
print (test_input)

# Get input and output of the model
input_index = input_details["index"]
output_index = interpreter.get_output_details()[0]["index"]

# Example of turning test input into predictions
interpreter.set_tensor(input_index, test_input)
interpreter.invoke()
predictions = interpreter.get_tensor(output_index)

# this will be an array, with each array index holding and array of the output values
print (predictions)

'''
# Test a whole csv file of inputs
filename = 'compiled_yellow.csv'
data_lines = np.genfromtxt(filename, delimiter=',', skip_header=1)
for i in range(len(data_lines)):
    test_input = np.array([data_lines[i]]).astype(np.float32)
    interpreter.set_tensor(input_index, test_input)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_index)
    if float(predictions[0][2]) < 0.98:
        print ("Data:" + str(data_lines[i]) + " " + str(predictions[0]))
'''
