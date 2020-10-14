import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
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
images, paths = utils.load_images_all()

# For each images
for i, image_yolo in enumerate(images):

    if i == 1:
        continue
    elif i == (len(images)-1):
        detections = wrapper.detect_image(network, ['Car'], images[i], thresh=.25)
        detections2 = wrapper.detect_image(network, ['Car'], images[0], thresh=.25) 
    else:
        detections = wrapper.detect_image(network, ['Car'], images[i], thresh=.25)
        detections2 = wrapper.detect_image(network, ['Car'], images[i], thresh=.25) 

    if (not detections) or (not detections2):
        print("One photo has no car, continuing...")
        continue

    # Get bbox best coordinates of the detections
    xmin, ymin, xmax, ymax, center = utils.retrieve_best_coordinates(detections, images[i])
    xmin2, ymin2, xmax2, ymax2, center2 = utils.retrieve_best_coordinates(detections2, images[i])

    # Load the images as Numpy narrays
    img = cv.imread(paths[i], cv.IMREAD_GRAYSCALE)
    img2 = cv.imread(paths[i], cv.IMREAD_GRAYSCALE)

    # Instantiate the KeyPoint class from the centers coordinates
    kp_center = cv.KeyPoint(center[0], center[1], 0)
    kp_center2 = cv.KeyPoint(center2[0], center2[1], 0)

    # Apply MSER+SIFT with L2 filter from the YOLO bbox centers
    (kp, des), (kp2, des2), good = utils.apply(img, img2, (xmin, ymin, xmax, ymax), (xmin2, ymin2, xmax2, ymax2), kp_center, kp_center2)

    #matches_kp1 = [kp[mat[0].trainIdx].pt for mat in good] 
    #matches_kp2 = [kp2[mat[0].queryIdx].pt for mat in good]

    print(len(kp))
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(len(kp2))
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    #print(good)
    #print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    img3 = cv.drawMatchesKnn(img,kp,img2,kp2,good,None, flags=0)
    plt.imshow(img3),plt.show()



