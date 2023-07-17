from ultralytics import YOLO

model = YOLO('yolov8m.pt')

model.train(data='FRC2023.yaml', epochs=300, imgsz=640, batch=8)