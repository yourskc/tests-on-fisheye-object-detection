# SPDX-License-Identifier: MIT
import time
import cv2
import numpy as np
import onnxruntime as ort
from box_utils import predict

# Constants for object detection
confidence_thres = 0.5
iou_thres = 0.5

# Load COCO classes
COCO_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush",
}

# Load models
def load_models():
    print("Loading models...")
    yolo_model = ort.InferenceSession("yolo_object_detection.onnx")
    return yolo_model

# Face detection utilities
def scale(box):
    width = box[2] - box[0]
    height = box[3] - box[1]
    maximum = max(width, height)
    dx = int((maximum - width)/2)
    dy = int((maximum - height)/2)
    bboxes = [box[0] - dx, box[1] - dy, box[2] + dx, box[3] + dy]
    return bboxes

# Object detection utilities
def nms_boxes(boxes, scores, score_threshold, nms_threshold):
    if len(boxes) == 0:
        return np.array([], dtype=np.int32)
    boxes = np.array(boxes)
    scores = np.array(scores)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = x1 + boxes[:, 2]
    y2 = y1 + boxes[:, 3]
    areas = boxes[:, 2] * boxes[:, 3]
    order = scores.argsort()[::-1]
    keep = []
    
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        intersection = w * h
        iou = intersection / (areas[i] + areas[order[1:]] - intersection)
        inds = np.where(iou <= nms_threshold)[0]
        order = order[inds + 1]
    
    return np.array(keep, dtype=np.int32)

def process_frame(frame, yolo_model):
    # Time tracking
    frame_time_start = time.time()
    
    # Get YOLO input shape
    model_inputs = yolo_model.get_inputs()
    input_width = model_inputs[0].shape[2]
    input_height = model_inputs[0].shape[3]
    
    # Object Detection
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_width, input_height))
    image_data = np.array(img) / 255.0
    image_data = image_data.transpose(2, 0, 1)
    image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
    
    # Run YOLO inference
    outputs = yolo_model.run(None, {model_inputs[0].name: image_data})
    outputs = np.transpose(np.squeeze(outputs[0]))
    
    # Process YOLO outputs
    rows = outputs.shape[0]
    boxes = []
    scores = []
    class_ids = []
    
    x_factor = frame.shape[1] / input_width
    y_factor = frame.shape[0] / input_height
    
    for i in range(rows):
        classes_scores = outputs[i][4:]
        max_score = np.amax(classes_scores)
        if max_score >= confidence_thres:
            class_id = np.argmax(classes_scores)
            x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
            left = int((x - w / 2) * x_factor)
            top = int((y - h / 2) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)
            
            class_ids.append(int(class_id))
            scores.append(float(max_score))
            boxes.append([left, top, width, height])
    
    # Draw object detection results
    if len(boxes) > 0:
        indices = nms_boxes(boxes, scores, confidence_thres, iou_thres)
        for i in indices:
            # if class_ids[i] == 41:  # cup
            if class_ids[i] < 80: # All  
                box = boxes[i]
                left, top = box[0], box[1]
                width, height = box[2], box[3]
                cv2.rectangle(frame, (left, top), (left + width, top + height), (0, 255, 0), 2)
                # label = f"cup: {scores[i]:.2f}"
                label = f"{COCO_CLASSES[class_ids[i]]}: {scores[i]:.2f}"
                cv2.putText(frame, label, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)    
    # Calculate FPS
    total_time = time.time() - frame_time_start
    fps = 1.0 / total_time if total_time > 0 else 0
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    return frame

def main():
    # Load models
    yolo_model = load_models()
    print("Models loaded successfully")
    
    # Setup video capture
    """
    pipeline = "v4l2src device=/dev/video0 ! videoconvert ! video/x-raw, width=640, height=480 ! appsink "    
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    
    print("Camera opened successfully")
    """

    # ret, frame = cap.read()
    frame = cv2.imread('camera1_A_1.png')  
        
    # Process frame            
    processed_frame = process_frame(frame, yolo_model)
            
    # Display results
    cv2.imshow('Detection Results', processed_frame)
    cv2.waitKey(0) 
    cv2.imwrite('result.png', processed_frame)    
   
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
