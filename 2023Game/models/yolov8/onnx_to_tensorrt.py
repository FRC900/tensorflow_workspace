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
 
def onnx_to_tensorrt(onnx_model,
                     output, 
                     max_workspace_size=1<<25,
                     int8=False,
                     fp16=False,
                     dla_core=None,
                     batch_size=1,
                     gpu_fallback=False,
                     dataset_path='datasets/FRC2023/images/train',
                     calibration_file='calib_FRC2023m.bin') -> None:
    # Get model input size from the onnx model data
    model = onnx.load(onnx_model)
    input_shapes = [[d.dim_value for d in _input.type.tensor_type.shape.dim] for _input in model.graph.input]
    print(input_shapes)
    model_channels = input_shapes[0][1]
    model_width = input_shapes[0][2]
    model_height = input_shapes[0][3]

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    builder.max_batch_size = batch_size
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    onnx_parser = trt.OnnxParser(network, logger)

    with open(onnx_model, 'rb') as f:
        onnx_parser.parse(f.read())

    profile = builder.create_optimization_profile()
    profile.set_shape(
        'input',
        (batch_size, model_channels, model_height, model_width),
        (batch_size, model_channels, model_height, model_width),
        (batch_size, model_channels, model_height, model_width),
    )

    config = builder.create_builder_config()

    config.max_workspace_size = max_workspace_size

    if fp16:
        print("Setting FP16 flag")
        config.set_flag(trt.BuilderFlag.FP16)

    if int8:
        from calibrator import YOLOEntropyCalibrator
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = YOLOEntropyCalibrator(dataset_path, 
                                                    (model_height, model_width),
                                                    calibration_file)
        config.set_calibration_profile(profile)
        print("Setting INT8 flag + calibrator")

    if gpu_fallback:
        print("Setting GPU_FALLBACK")
        config.set_flag(trt.BuilderFlag.GPU_FALLBACK)

    if dla_core is not None:
        config.default_device_type = trt.DeviceType.DLA
        config.DLA_core = dla_core
        config.set_flag(trt.BuilderFlag.STRICT_TYPES)
        print(f'Using DLA core {dla_core}')

    config.add_optimization_profile(profile)
    config.set_calibration_profile(profile)

    engine = builder.build_serialized_network(network, config)

    if output is not None:
        with open(output, 'wb') as f:
            f.write(engine)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('onnx', type=str, help='Path to the ONNX model.')
    parser.add_argument('--output', type=str, default=None, help='Path to output the optimized TensorRT engine')
    parser.add_argument('--max-workspace-size', type=int, default=1<<25, help='Max workspace size for TensorRT engine.')
    parser.add_argument('--int8', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--dla-core', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--gpu-fallback', action='store_true')
    parser.add_argument('--dataset-path', type=str, default='datasets/FRC2023/images/train')
    parser.add_argument('--calibration-file', type=str, default='calib_FRC2023m.bin')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    onnx_to_tensorrt(args.onnx,
                     args.output,
                     args.max_workspace_size,
                     args.int8,
                     args.fp16,
                     args.dla_core,
                     args.batch_size,
                     args.gpu_fallback,
                     args.dataset_path,
                     args.calibration_file)