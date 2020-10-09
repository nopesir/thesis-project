import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import json
import subprocess
import os

import wrapper


# Darknet configuration
config_file = "./darknet/cfg/yolov4-thesis.cfg"
data_file = "./darknet/thesis.data"
weights_file = "./darknet/yolov4-thesis_last.weights"

# Load image for processing
image = wrapper.load_image(b'./4.jpg', 0, 0)

# Load network and weights
network, class_names, colors = wrapper.load_network(config_file, data_file, weights_file)

# Perform the detection
detections = wrapper.detect_image(network, ['Car'], image, thresh=.25) 
detections = sorted(detections, key=lambda x: float(x[1]))
print(detections)
print(image.w)
print(image.h)
print(wrapper.bbox2points(detections[len(detections)-1][2]))
xmin, ymin, xmax, ymax = wrapper.bbox2points(detections[len(detections)-1][2])
if xmin < 0:
    xmin = 0

if xmax > image.w:
    xmax = image.w

if ymin < 0:
    ymin = 0

if ymax > image.h:
    ymax = image.h

print(xmin, ymin, xmax, ymax)
center = (int((xmax+xmin)/2), int((ymax+ymin)/2))
print(center)
# if(left < 0) left = 0; if(right > im.w-1) right = im.w-1; if(top < 0) top = 0; if(bot > im.h-1) bot = im.h-1;


img = cv.imread('4.jpg', 0)

# Initiate ORB detector
orb = cv.ORB_create()

# find the keypoints with ORB
kp = orb.detect(img,None)

# compute the descriptors with ORB
kp, des = orb.compute(img, kp)

# Keypoint of the center
kp2 = cv.KeyPoint(center[0], center[1], 0)

img_box = wrapper.draw_boxes(detections, img, colors)
plt.imshow(img_box), plt.show()

# Keypoints into detected object
kp_yolo = []

for i, keypoint in enumerate(kp):
    if (keypoint.pt[0] >= xmin) and (keypoint.pt[0] <= xmax) and (keypoint.pt[1] >= ymin)  and (keypoint.pt[1] <= ymax):
        kp_yolo.append(keypoint)

#for keypoint in kp:
#    print(keypoint.pt)


#img4 = cv.drawKeypoints(img, kp_yolo, None, color=(0,200,0), flags=0)
#plt.imshow(img4), plt.show()

kp_yolo.sort(key = lambda p: (p.pt[0] - kp2.pt[0])**2 + (p.pt[1] - kp2.pt[1])**2)


kp_yolo = kp_yolo[:20]
img = cv.imread('4.jpg', 0)
img2 = cv.drawKeypoints(img, kp_yolo, None, color=(255,0,0), flags=0)
plt.imshow(img2), plt.show()

