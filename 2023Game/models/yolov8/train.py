import argparse
from ultralytics import YOLO
from pathlib import Path
from onnx_to_tensorrt import onnx_to_tensorrt


def train_yolo(args: argparse.Namespace) -> None:
    model = YOLO(args.yolo_model)


    print(f"{args.input_shape}")
    if isinstance(args.input_shape, int):
        imgsz = args.input_shape
    elif isinstance(args.input_shape, list):
        if len(args.input_shape) == 1:
            imgsz = args.input_shape[0]
        else:
            imgsz = (args.input_shape[0], args.input_shape[1])


    pt_file_path = model.train(data=args.config,
                               epochs=args.epochs,
                               imgsz=imgsz,
                               batch=args.batch_size)

    # Now convert from pytorch .pt format to a .onnx file
    # This is an intermediate step - the onnx file format is generic,
    # and in this case provides a way to translate from .pt to
    # an optimized TensorRT engine file

    import subprocess
    export_det_args = [
        'python3',
        '/home/ubuntu/YOLOv8-TensorRT/export-det.py',
        '--weights', pt_file_path,
        '--iou-thres', '0.65',
        '--conf-thres', '0.25',
        '--topk', '100',
        '--opset', '11',
        '--sim',
        '--input-shape', '1',  '3', '640', '640', 
        '--device', 'cuda:0',
    ]
    subprocess.run(export_det_args)


    # The output will be a file with .onnx as the extension
    onnx_path = Path(pt_file_path).with_suffix('.onnx')

    # Create a tensorrt .engine file. This is a model optimized for the specific
    # hardware we're currently running on. That will be useful for testing
    # the model locally.
    # Additionally, it will create a calibration file useful for optimizing
    # int8 models on other platforms.  
    tensorrt_path = onnx_path.with_name(Path(pt_file_path).stem + '_int8').with_suffix('.engine')
    calibration_path = onnx_path.with_name('calib_FRC2023m.bin')

    onnx_to_tensorrt(onnx_path,
                     tensorrt_path,
                     int8=True,
                     fp16=True,
                     dataset_path='datasets/FRC2023/images/train',
                     calibration_file=calibration_path)

    print(f"Training finished, generated {pt_file_path}, {onnx_path}, {tensorrt_path} and {calibration_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-model',
                        type=str,
                        default='yolov8m.pt',
                        help='Engine file')
    parser.add_argument('--config',
                        type=str,
                        default='FRC2023.yaml',
                        help='Config file')
    parser.add_argument('--epochs',
                        type=int,
                        default=200,
                        help='Number of epochs to train')
    parser.add_argument('--batch-size',
                        type=int,
                        default=8,
                        help='Number of images to batch')
    parser.add_argument('--input-shape',
                        nargs='+',
                        type=int,
                        default=[640, 640],
                        help='Model input shape')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    train_yolo(args)