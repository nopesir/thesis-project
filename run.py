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
image = wrapper.load_image(b'./1.jpg', 0, 0)

# Load network and weights
network, class_names, colors = wrapper.load_network(config_file, data_file, weights_file)

# Perform the detection
detections = wrapper.detect_image(network, ['Car'], image) 

xmin, ymin, xmax, ymax = wrapper.bbox2points(detections[0][2])
center = ((xmax-xmin)/2, (ymax-ymin)/2)

'''
with open('results.json') as json_file:
    data = json.load(json_file)

print(data)

'''

img = cv.imread('1.jpg', 0)

# Initiate ORB detector
orb = cv.ORB_create()

# find the keypoints with ORB
kp = orb.detect(img,None)

# compute the descriptors with ORB
kp, des = orb.compute(img, kp)



img2 = cv.drawKeypoints(img, kp, None, color=(255,0,0), flags=0)
plt.imshow(img2), plt.show()

for i,keypoint in enumerate(kp):
    if xmin <= keypoint.pt[0] and keypoint.pt[0] <= xmax and ymin <= keypoint.pt[1] and keypoint.pt[1] <= ymax:
        continue 
    else:
        kp.remove(keypoint)


kp.sort(key = lambda p: (p.pt[0] - center.pt[0])**2 + (p.pt[1] - center.pt[1])**2)
print(len(kp))
todel = len(kp)-5
kp = kp[:10]
print(len(kp))
img2 = cv.drawKeypoints(img, kp, None, color=(255,0,0), flags=0)
plt.imshow(img2), plt.show()

