# ref. https://blog.csdn.net/qq_41204464/article/details/142942825
from ultralytics import YOLO
model = YOLO('yolo_object_detection.pt')
model.export(
    format='onnx',
    imgsz=(640,640),
    keras=False,
    optimize=False,
    half=False,
    int8=False,
    dynamic=False,
    simplify=False,
    opset=None,
    workspace=4.0,
    nms=False,
    batch=1,
    device="cpu"
)

