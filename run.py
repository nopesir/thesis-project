import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import json
import subprocess
import os

import wrapper


image = wrapper.load_image(b"1.jpg", 0,0)

print(wrapper.detect_image(wrapper.load_network("./darknet/cfg/yolov4-thesis.cfg", "./darknet/thesis.data", "./darknet/yolov4-thesis_last.weights"), ['Car'], image))

with open('results.json') as json_file:
    data = json.load(json_file)

print(data)


img = cv.imread('1.jpg', 0)
# Initiate ORB detector
orb = cv.ORB_create()
# find the keypoints with ORB
kp = orb.detect(img,None)
# compute the descriptors with ORB
kp, des = orb.compute(img, kp)
# draw only keypoints location,not size and orientation
for i,keypoint in enumerate(kp):
    print("Keypoint %d: %f %f" % (i, keypoint.pt[0]/2560, keypoint.pt[1]/1600))