# Fast(er) RCNN

Toy implementation of Fast RCNN, with ideas of expanding into Faster RCNN

### Currently using
  - OpenCV Selective Search
  - MobileNet with ImageNet weights

Idea of expanding into using RPN and ResNet/Vgg16

### Steps
1. First, run create_dataset.py, using:
``` python create_dataset.py```
This will take the images in raccoons/ and create a dataset for fine tuning the MobileNet model

2. Run fine_tune.py
``` python fine_tune.py```
This will fine tune the MobileNet model using the dataset created in step 1

3. Run detect_object.py passing a raccoon image as parameter
``` python detect_object_rcnn.py -i images/raccoon_01.jpg``` 
This will output the model predictions for the image
