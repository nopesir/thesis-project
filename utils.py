"""
Python 3 utils functions for thesis-project
Use it as Python module:

import utils 

utils.function()
"""



import wrapper
import os
import cv2 as cv


def retrieve_best_coordinates(detections, image_yolo):
    """
    Get the coordinates of the best detection as (xmin, ymin, xmax, ymax, center)
    """
    xmin, ymin, xmax, ymax = wrapper.bbox2points(detections[len(detections)-1][2])
    
    if xmin < 0:
        xmin = 0

    if xmax > image_yolo.w:
        xmax = image_yolo.w

    if ymin < 0:
        ymin = 0

    if ymax > image_yolo.h:
        ymax = image_yolo.h

    center = (int((xmax+xmin)/2), int((ymax+ymin)/2))

    return xmin, ymin, xmax, ymax, center


def kp_filtersort_L2(kp, bbox, kp_center, n=20):
    """
    Filter out the keypoints not in the bbox and discards the ALL n-ones that are far from the bbox center
    """

    kp_yolo = []

    for keypoint in kp:
        if (keypoint.pt[0] >= bbox[0]) and (keypoint.pt[0] <= bbox[2]) and (keypoint.pt[1] >= bbox[1])  and (keypoint.pt[1] <= bbox[3]):
            kp_yolo.append(keypoint)
    


    kp_yolo.sort(key = lambda p: (p.pt[0] - kp_center.pt[0])**2 + (p.pt[1] - kp_center.pt[1])**2)


    kp_yolo = kp_yolo[:n]

    return kp_yolo


def apply(img1, img2, bbox1, bbox2, kp_center1, kp_center2):
    """
    Apply MSER+SIFT on the bbox of the two images and filter the keypoints using L2 NORM distance from the YOLO bbox center.
    """

    # Initiate MSER detector
    mser = cv.MSER_create()
    # Initiate SIFT descriptor
    sift = cv.SIFT_create()

    # Find the keypoints with MSER
    kp = mser.detect(img1, None)

    # Filter only the first n points closest to the YOLO bbox center
    kp = kp_filtersort_L2(kp, bbox1, kp_center1, n=50)

    # Compute the descriptors with SIFT
    kp, des = sift.compute(img1, kp)

    # Find the keypoints with MSER
    kp2 = mser.detect(img2, None)

    # Filter only the first n points closest to the YOLO bbox center
    kp2 = kp_filtersort_L2(kp2, bbox2, kp_center2, n=50)

    # Compute the descriptors with SIFT
    kp2, des2 = sift.compute(img2, kp2)

    # Brute Force matcher with default params (L2_NORM)
    bf = cv.BFMatcher()

    # Match using KNN with k=2
    matches = bf.knnMatch(des, des2, k=2)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])

    return (kp, des), (kp2, des2), good


def load_images():

    """
    Load all YOLO IMAGE class file from the image.txt file into a list and return it
    """

    images = []
    paths = []
    with open("images.txt") as f_images:
        temp = f_images.readlines()
        for i, line in enumerate(temp):
            paths.insert(i, line.replace('\n', ''))

    for line in paths:
        images.append(wrapper.load_image(bytes(line, encoding='utf-8'), 0, 0))
    
    return images, paths


def load_images_all():
    """
    Load all YOLO IMAGE class file in the folder into a list and return it
    """

    images = []
    paths = []

    for i, file in enumerate([f for f in os.listdir('./images/') if f.lower().endswith(('.png', '.jpg', '.jpeg'))]):
        paths.insert(i, "images/" + file)
        images.append(wrapper.load_image(bytes(paths[i], encoding='utf-8'), 0, 0))
        
    
    return images, paths