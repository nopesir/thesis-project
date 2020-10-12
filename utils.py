"""
Python 3 utils functions for thesis-project
Use it as Python module:

import utils 

utils.function()
"""



import wrapper
import os
import cv2 as cv
import math


def ssc(keypoints, cols, rows, num_ret_points=20, tolerance=.4):
    exp1 = rows + cols + 2 * num_ret_points
    exp2 = (4 * cols + 4 * num_ret_points + 4 * rows * num_ret_points + rows * rows + cols * cols -
            2 * rows * cols + 4 * rows * cols * num_ret_points)
    exp3 = math.sqrt(exp2)
    exp4 = num_ret_points - 1

    sol1 = -round(float(exp1 + exp3) / exp4)  # first solution
    sol2 = -round(float(exp1 - exp3) / exp4)  # second solution

    high = sol1 if (sol1 > sol2) else sol2  # binary search range initialization with positive solution
    low = math.floor(math.sqrt(len(keypoints) / num_ret_points))

    prev_width = -1
    selected_keypoints = []
    result_list = []
    result = []
    complete = False
    k = num_ret_points
    k_min = round(k - (k * tolerance))
    k_max = round(k + (k * tolerance))

    while not complete:
        width = low + (high - low) / 2
        if width == prev_width or low > high:  # needed to reassure the same radius is not repeated again
            result_list = result  # return the keypoints from the previous iteration
            break

        c = width / 2  # initializing Grid
        num_cell_cols = int(math.floor(cols / c))
        num_cell_rows = int(math.floor(rows / c))
        covered_vec = [[False for _ in range(num_cell_cols + 1)] for _ in range(num_cell_rows + 1)]
        result = []

        for i in range(len(keypoints)):
            row = int(math.floor(keypoints[i].pt[1] / c))  # get position of the cell current point is located at
            col = int(math.floor(keypoints[i].pt[0] / c))
            if not covered_vec[row][col]:  # if the cell is not covered
                result.append(i)
                # get range which current radius is covering
                row_min = int((row - math.floor(width / c)) if ((row - math.floor(width / c)) >= 0) else 0)
                row_max = int(
                    (row + math.floor(width / c)) if (
                            (row + math.floor(width / c)) <= num_cell_rows) else num_cell_rows)
                col_min = int((col - math.floor(width / c)) if ((col - math.floor(width / c)) >= 0) else 0)
                col_max = int(
                    (col + math.floor(width / c)) if (
                            (col + math.floor(width / c)) <= num_cell_cols) else num_cell_cols)
                for row_to_cover in range(row_min, row_max + 1):
                    for col_to_cover in range(col_min, col_max + 1):
                        if not covered_vec[row_to_cover][col_to_cover]:
                            # cover cells within the square bounding box with width w
                            covered_vec[row_to_cover][col_to_cover] = True

        if k_min <= len(result) <= k_max:  # solution found
            result_list = result
            complete = True
        elif len(result) < k_min:
            high = width - 1  # update binary search range
        else:
            low = width + 1
        prev_width = width

    for i in range(len(result_list)):
        selected_keypoints.append(keypoints[result_list[i]])

    return selected_keypoints

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


def kp_filtersort_L2(kp, img, bbox, kp_center, n=100):
    """
    Filter out the keypoints not in the bbox and discards the ALL n-ones that are far from the bbox center
    """




    kp = ssc(kp, img.shape[1], img.shape[0], num_ret_points=n)

    kp.sort(key = lambda p: (p.pt[0] - kp_center.pt[0])**2 + (p.pt[1] - kp_center.pt[1])**2)

    kp = kp[:5]

    kp_yolo = []

    for keypoint in kp:
        if (keypoint.pt[0] >= bbox[0]) and (keypoint.pt[0] <= bbox[2]) and (keypoint.pt[1] >= bbox[1])  and (keypoint.pt[1] <= bbox[3]):
            kp_yolo.append(keypoint)



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
    kp = kp_filtersort_L2(kp, img1, bbox1, kp_center1, n=int(len(kp)/10))

    # Compute the descriptors with SIFT
    kp, des = sift.compute(img1, kp)

    # Find the keypoints with MSER
    kp2 = mser.detect(img2, None)

    # Filter only the first n points closest to the YOLO bbox center
    kp2 = kp_filtersort_L2(kp2, img2, bbox2, kp_center2, n=int(len(kp2)/10))

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