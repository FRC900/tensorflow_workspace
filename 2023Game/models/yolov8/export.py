from ultralytics import YOLO
import torch # Needed for model.export(engine) to find GPU?

# TODO - make command line arg?
model = YOLO('/home/ubuntu/tensorflow_workspace/2023Game/models/yolov8/runs/detect/train6/weights/best.pt')

# Need to run on Jetson so it knows which GPU to optimize for
# TODO - turn off verbose mode
engine_path = model.export(format='engine', device=0, imgsz=640, half=True, simplify=True)

# PyTorch: starting from /home/ubuntu/tensorflow_workspace/2023Game/models/yolov8/runs/detect/train6/weights/best.pt with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 28, 8400) (49.6 MB)


##python3 /home/ubuntu/YOLOv8-TensorRT/export-det.py --weights runs/detect/train6/weights/best.pt --iou-thres 0.65 --conf-thres 0.25 --topk 100 --opset 11 --sim --input-shape 1 3 640 640 --device cuda:0

##/usr/src/tensorrt/bin/trtexec --onnx=./runs/detect/train6/weights/best.onnx --saveEngine=./runs/detect/train6/weights/best.engine --fp16 --memPoolSize=13000