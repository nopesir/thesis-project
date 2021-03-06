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



#matches = utils.run_superglue(pairs_folder, network, images)
matches = utils.run_surf(images, network)


if not matches:
    print("No common matches found, decrease the number of photos")
    sys.exit(0)

common_kps = utils.retrieve_common_kps(matches)
for i in common_kps:
    print(i)

axis = np.float32([[580,360,50], [680,360,50], [680,460,50], [580,460,50],
                   [580,360,-50], [680,360,-50], [680,460,-50], [580,460,-50]])

kps = []
for i, kp in enumerate(common_kps[0]):
    kps.append(cv.KeyPoint(kp[0], kp[1], 0))
    common_kps[0][i] = list(kp)

img = cv.imread(images[0][1], cv.IMREAD_GRAYSCALE)
last = cv.drawKeypoints(img, kps,None)
plt.imshow(last),plt.show()


# Create the 3D points from the common keypoints (Z=1 for planar objects)
imgp = np.array(common_kps[0], np.float32)
objp = np.ones((len(imgp), 3), np.float32)

for i,elem in enumerate(objp):
    for k in range(0, 2):
        if k!=2:
            objp[i][k] = imgp[i][k] 

for i, elem in enumerate(common_kps):
    imgp = np.array(elem, np.float32)
    img = cv.imread(images[i][1])
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    imgp = np.reshape(imgp, (len(imgp), 1, 2))
    ret, rvecs, tvecs, _ = cv.solvePnPRansac(objp, imgp, K, d)
    print(ret)
    R = cv.Rodrigues(rvecs)[0]
    C = -R.T.dot(tvecs)
    print("Camera #" + str(i+1) + " position: " + str(C.ravel()))
    # project 3D points to image plane
    imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, K, d)

    img = utils.draw(img,imgp,imgpts)
    plt.imshow(img), plt.show()

sys.exit(0)
