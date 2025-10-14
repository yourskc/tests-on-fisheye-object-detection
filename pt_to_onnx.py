from ultralytics import YOLO
model = YOLO('yolo_object_detection.pt')
model.export(format='onnx')
