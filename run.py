import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os
import sys
import utils
import wrapper
import itertools
from config import *


# Load network and weights
network, class_names, colors = wrapper.load_network(config_file, data_file, weights_file)

# Load images and correstonding paths from image.txt file
images = utils.load_images(images_file)

# Initialization
matches = []
k = 0

# For each images
for i, image_yolo in enumerate(images):
    first = 0
    second = 0
    if i>=(len(images)-1):
        continue
    else:
        second = i+1

    detections = wrapper.detect_image(network, ['Car'], images[first][0], thresh=.15)
    detections2 = wrapper.detect_image(network, ['Car'], images[second][0], thresh=.15)   

    if (not detections) or (not detections2):
        print("nope: " + images[first][1] + " " + images[second][1])
        continue
    

    # Get bbox best coordinates of the detections
    bbox, center = utils.retrieve_best_coordinates(detections, images[first][0])
    bbox2, center2 = utils.retrieve_best_coordinates(detections2, images[second][0])


    # Load the images as Numpy narrays
    img = cv.imread(images[first][1], cv.IMREAD_GRAYSCALE)
    img2 = cv.imread(images[second][1], cv.IMREAD_GRAYSCALE)

    # Instantiate the KeyPoint class from the centers bboxes coordinates
    kp_center = cv.KeyPoint(center[0], center[1], 0)
    kp_center2 = cv.KeyPoint(center2[0], center2[1], 0)
    
    # Apply SURF with L2 filter from the YOLO bbox centers
    (kp, des), (kp2, des2), good = utils.apply(img, img2, bbox, bbox2, kp_center, kp_center2)

    matches_kp1 = []
    matches_kp2 = []

    for match in good:
        matches_kp1.append(kp[match[0].trainIdx].pt) 
        matches_kp2.append(kp[match[0].queryIdx].pt) 

    if k==0:
        matches.append((matches_kp1, matches_kp2))

    #print("\n++++ Matches for " + images[first][1] + images[second][1] + " ++++")
    #for i,kpp in enumerate(matches_kp1):
    #    print("[" + str(kpp[0]) + " , " + str(kpp[1]) + "] --> [" + str(matches_kp2[i][0]) + " , " + str(matches_kp2[i][1]) +  "]") 



    img3 = cv.drawMatchesKnn(img,kp,img2,kp2,good,None, flags=2)
    plt.imshow(img3),plt.show()

print(matches)
common_kps = utils.retrieve_common_kps(matches)
common_kps.pop()
axis = np.float32([[150,0,0], [0,150,0], [0,0,-150]]).reshape(-1,3)

kps = []
for i, kp in enumerate(common_kps):
    kps.append(cv.KeyPoint(kp[0], kp[1], 0))
    common_kps[i] = list(kp)

last = cv.drawKeypoints(img, kps,None)
plt.imshow(last),plt.show()


# Create the 3D points from the common keypoints (Z=0 for planar objects)
imgp = np.array(common_kps)
objp = np.zeros((len(imgp), 3), np.float32)

for i,elem in enumerate(objp):
    for k in range(0, 2):
        if k!=2:
            objp[i][k] = imgp[i][k] 

img = cv.imread(images[0][1])
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

imgp = np.reshape(imgp, (len(imgp), 1, 2))
print(imgp.shape)

#corners2 = cv.cornerSubPix(gray,imgp,(11,11),(-1,-1),criteria)
ret, rvecs, tvecs, inliers = cv.solvePnPRansac(objp, imgp, K, None)

# project 3D points to image plane
imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, K, None)

ttt = cv.drawChessboardCorners(img, (3,6), imgp, ret)

img = utils.draw(img,imgp,imgpts)


sys.exit(0)
