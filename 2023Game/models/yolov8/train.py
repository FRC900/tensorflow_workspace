import argparse
from ultralytics import YOLO
from os import rename

def train_yolo(args: argparse.Namespace) -> None:
    model = YOLO(args.yolo_model)

    pt_file_path = model.train(data=args.config,
                               epochs=args.epochs,
                               imgsz=args.input_size,
                               batch=args.batch_size)

    # Now convert from pytorch .pt format to a .onnx file
    # This is an intermediate step - the onnx file format is generic,
    # and in this case provides a way to translate from .pt to
    # an optimized TensorRT engine file

    # For testing :  pt_file_path = 'runs/detect/train2/weights/best.pt'
    from pathlib import Path
    # Need to import this after caling YOLO or training fails?
    from onnx_to_tensorrt import onnx_to_tensorrt
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
        '--input-shape', '1',  '3', f'{args.input_size}', f'{args.input_size}', 
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
    tensorrt_path = onnx_path.with_name(args.output_stem + '_int8').with_suffix('.engine')
    calibration_path = onnx_path.with_name('calib_' + args.output_stem + '.bin')

    onnx_to_tensorrt(onnx_path,
                     tensorrt_path,
                     int8=True,
                     fp16=True,
                     dataset_path='datasets/FRC2023/images/train',
                     calibration_file=calibration_path)
    new_pt_file_path = Path(pt_file_path).with_stem(args.output_stem)
    new_onnx_path = onnx_path.with_stem(args.output_stem)
    rename(pt_file_path, new_pt_file_path)
    rename(onnx_path, new_onnx_path)
    print(f"Training finished, generated {new_pt_file_path}, {new_onnx_path}, {tensorrt_path} and {calibration_path}")


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
    parser.add_argument('--output_stem',
                        type=str,
                        default='best',
                        help='File name stem for pt, onnx, engine and calibration file')
    parser.add_argument('--epochs',
                        type=int,
                        default=200,
                        help='Number of epochs to train')
    parser.add_argument('--batch-size',
                        type=int,
                        default=8,
                        help='Number of images to batch')
    parser.add_argument('--input-size',
                        type=int,
                        default=640,
                        help='Model input image size')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    train_yolo(args)
