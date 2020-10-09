import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import json
import subprocess
import os
import sys
import utils

import wrapper


# Darknet configuration
config_file = "./darknet/cfg/yolov4-thesis.cfg"
data_file = "./darknet/thesis.so.data"
weights_file = "./darknet/yolov4-thesis_last.weights"

# Load network and weights
network, class_names, colors = wrapper.load_network(config_file, data_file, weights_file)

# Load images and correstonding paths from image.txt file
images, paths = utils.load_images()

# For each images
for i, image_yolo in enumerate(images):

    # Perform the detections
    detections = wrapper.detect_image(network, ['Car'], image_yolo, thresh=.25) 

    # Get bbox best coordinates of the detection
    xmin, ymin, xmax, ymax, center = utils.retrieve_best_coordinates(detections, image_yolo)

    img = cv.imread(paths[i], 0)
    kp_center = cv.KeyPoint(center[0], center[1], 0)

    kp_yolo = utils.apply_yolo_orb(img, (xmin, ymin, xmax, ymax), kp_center)


    img = cv.imread(paths[i], 0)
    img2 = cv.drawKeypoints(img, kp_yolo, None, color=(255,0,0), flags=0)
    img2 = cv.drawKeypoints(img2, [kp_center], None, color=(0,255,0), flags=0)
    plt.imshow(img2), plt.show()


