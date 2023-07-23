# Adapted from https://raw.githubusercontent.com/NVIDIA-AI-IOT/jetson_dla_tutorial/master/build.py
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

# Takes an onnx model as input, outputs an optmized tensorrt engine
# This needs to be run on the target GPU (e.g. on the jetson) since
# the optimization is based on timing info obtained by running the 
# model on actual hardware
import argparse
import tensorrt as trt
import pycuda.autoinit
import onnx

parser = argparse.ArgumentParser()
parser.add_argument('onnx', type=str, help='Path to the ONNX model.')
parser.add_argument('--output', type=str, default=None, help='Path to output the optimized TensorRT engine')
parser.add_argument('--max_workspace_size', type=int, default=1<<25, help='Max workspace size for TensorRT engine.')
parser.add_argument('--int8', action='store_true')
parser.add_argument('--fp16', action='store_true')
parser.add_argument('--dla_core', type=int, default=None)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--gpu_fallback', action='store_true')
parser.add_argument('--dataset_path', type=str, default='datasets/FRC2023/images/train')
parser.add_argument('--calibration_file', type=str, default='calib_FRC2023m.bin')
args = parser.parse_args()
 
# Get model input size from the onnx model data
model = onnx.load(args.onnx)
input_shapes = [[d.dim_value for d in _input.type.tensor_type.shape.dim] for _input in model.graph.input]
print(input_shapes)
model_channels = input_shapes[0][1]
model_width = input_shapes[0][2]
model_height = input_shapes[0][3]

logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
builder.max_batch_size = args.batch_size
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
onnx_parser = trt.OnnxParser(network, logger)

with open(args.onnx, 'rb') as f:
    onnx_parser.parse(f.read())

profile = builder.create_optimization_profile()
profile.set_shape(
    'input',
    (args.batch_size, model_channels, model_height, model_width),
    (args.batch_size, model_channels, model_height, model_width),
    (args.batch_size, model_channels, model_height, model_width),
)

config = builder.create_builder_config()

config.max_workspace_size = args.max_workspace_size

if args.fp16:
    print("Setting FP16 flag")
    config.set_flag(trt.BuilderFlag.FP16)

if args.int8:
    from calibrator import YOLOEntropyCalibrator
    config.set_flag(trt.BuilderFlag.INT8)
    config.int8_calibrator = YOLOEntropyCalibrator(args.dataset_path, 
                                                   (model_height, model_width),
                                                   args.calibration_file)
    config.set_calibration_profile(profile)
    print("Setting INT8 flag + calibrator")

if args.gpu_fallback:
    print("Setting GPU_FALLBACK")
    config.set_flag(trt.BuilderFlag.GPU_FALLBACK)

if args.dla_core is not None:
    print("Setting DLA core")
    config.default_device_type = trt.DeviceType.DLA
    config.DLA_core = args.dla_core
    config.set_flag(trt.BuilderFlag.STRICT_TYPES)
    print('Using DLA core %d.' % args.dla_core)

config.add_optimization_profile(profile)
config.set_calibration_profile(profile)

engine = builder.build_serialized_network(network, config)

if args.output is not None:
    with open(args.output, 'wb') as f:
        f.write(engine)
