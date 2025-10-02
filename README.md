# Tests on Fisheye Object Detection

We tested a model for fisheye object detection with fisheye images from another fisheye dataset 

### Fisheye Object Detection Model

https://github.com/arda92a/fisheye-object-detection

### Fisheye Image Dataset 

https://github.com/MoyoG/FishEye8K

The fisheye-object-detection project provides a Flask-based web application for object detection and segmentation in fisheye camera images and videos. It supports multiple deep learning models, including YOLO (for object detection and segmentation), RetinaNet, and Faster R-CNN, to detect and classify objects such as bikes, buses, cars, pedestrians, and trucks.



Follow the instructions in the repo' README.md to install dependencies, prepare
model and run the APP,

1. Install Dependencies: 
```
pip install -r requirements.txt
```
2. Prepare Model Files: 

[Download Link](https://drive.google.com/drive/folders/1f5sj08ztjm7DjV1sYxSHa-dVm-vnA56z?usp=sharing) 

We'll use download yolo_object_detection.pt (121M) and put in ./models folder.

3. Run the Application:
```
python app.py
```
Open the web browser, input the url,
```
http://localhost:5000
```

![](images/p01.png)

Select the Model,

Upload a fisheye image, here we use the image from the below dataset 

Dataset :

Fisheye image dataset with cars and bikes can be found here,

https://github.com/MoyoG/FishEye8K

![](images/p02.png)

dataset URL is as below,

https://scidm.nchc.org.tw/zh_TW/dataset/fisheye8k/resource/f6e7500d-1d6d-48ea-9d38-c4001a17170e

example dataset image,

![](images/p03.png)

Inference Result from the above APP's web interface: 

![](images/p04.png)

Similarily, we can open a terminal, and directly use yolo command to perform the inference, 

```
yolo predict model=yolo_object_detection.pt source="camera1_A_1.png"
```

Note: if you didn't installed 
```
pip install ultralytics
```
The result is
as below,

![](images/p05.png)


### Test the fisheye image using normal YOLO model

The normal YOLO model will detect objects for 
normal image, but can't detect object for fisheye image.

for normal image,

![](images/p06.png)

However, for fisheye image,

![](images/p03.png)

Nothing has been detected in the above fisheye image.

### Night sight - 1/2

This model can be used to detect object from night vision image well.

The original image is as below,

![](images/p07-1.png)

As we can see in the below, alrough the image has poor contrast,  the cars, motorcycles and pedestrians can be detected well. 

![](images/p07-2.png)

### Night sight - 2/2

The original image is as below,

![](images/p08-1.png)

The blurred motocycles at night can also be detected well. 

![](images/p08-2.png)