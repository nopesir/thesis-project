"""
Python 3 utils functions for thesis-project
Use it as Python module:

import utils 

utils.function()
"""
from __future__ import absolute_import, division, print_function
import hashlib
import zipfile
from six.moves import urllib
import cv2 as cv
from matplotlib import pyplot as plt
import wrapper
import os
import cv2 as cv
import math
import numpy as np
import glob

from config import *


def ssc(keypoints, cols, rows, num_ret_points=100, tolerance=.3):
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

    return (xmin, ymin, xmax, ymax), center

def kp_filtersort_L2(kp, img, bbox, kp_center, n=40):
    """
    Filter out the keypoints not in the bbox and discards the ALL n-ones that are far from the bbox center
    """

    #kp_test = ssc(kp, img.shape[1], img.shape[0], num_ret_points=500)
    kp_yolo = []
    for keypoint in kp:
        if (keypoint.pt[0] >= bbox[0]) and (keypoint.pt[0] <= bbox[2]) and (keypoint.pt[1] >= bbox[1])  and (keypoint.pt[1] <= bbox[3]):
            kp_yolo.append(keypoint)
    
    
    kp_yolo.sort(key = lambda p: (p.pt[0] - kp_center.pt[0])**2 + (p.pt[1] - kp_center.pt[1])**2)


    return kp_yolo[0:n]

def apply_gpu(img1, img2, bbox1, bbox2, kp_center1, kp_center2):
    """
    Still in development
    """

    cuMat1 = cv.cuda_GpuMat(img1)
    cuMat2 = cv.cuda_GpuMat(img2)


    c_surf = cv.cuda.SURF_CUDA_create(500)

    kp = c_surf.detect(cuMat1, None)
    kp = c_surf.downloadKeypoints(kp)
    kp = kp_filtersort_L2(kp, img1, bbox1, kp_center1)
    kp, des = c_surf.detectWithDescriptors(cuMat1, None,cv.cuda_GpuMat(kp))

    kp2 = c_surf.detect(cuMat2, None)
    kp2 = c_surf.downloadKeypoints(kp2)
    kp2 = kp_filtersort_L2(kp2, img2, bbox2, kp_center2)
    kp2, des2 = c_surf.compute(img2, kp2)
    
    
    # Brute Force matcher with default params (L2_NORM)
    cbf = cv.cuda_DescriptorMatcher.createBFMatcher(cv.NORM_L1)
    cmatches = cbf.match(des, des2) 

    # Sort matches by score
    cmatches.sort(key=lambda x: x.distance, reverse=False)
    # Remove not so good matches
    numGoodMatches = int(len(cmatches) * 0.15)
    cmatches = cmatches[:numGoodMatches] 

    return (kp, des), (kp2, des2), cmatches

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    corner = tuple([int(x) for x in corner])
    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

def apply(img1, img2, bbox1, bbox2, kp_center1, kp_center2):
    """
    Apply SURF on the bbox of the two images and filter the keypoints using L2 NORM distance from the YOLO bbox center.
    """

    surf = cv.xfeatures2d.SURF_create(350)

    kp = surf.detect(img1, None)
    kp = kp_filtersort_L2(kp, img1, bbox1, kp_center1)
    kp, des = surf.compute(img1, kp)

    kp2 = surf.detect(img2, None)
    kp2 = kp_filtersort_L2(kp2, img2, bbox2, kp_center2)
    kp2, des2 = surf.compute(img2, kp2)

    bf = cv.BFMatcher(normType=cv.NORM_L1)
    matches = bf.knnMatch(des, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.85 * n.distance:
            good.append([m])
    
    return (kp, des), (kp2, des2), good

def load_images(file):

    """
    Load all YOLO IMAGE class file from the image.txt file into a list and return it
    """

    images = []
    paths = readlines(file)

    for line in paths:
        images.append((wrapper.load_image(bytes(line, encoding='utf-8'), 0, 0),line))
    
    return images

def load_images_all(images_folder):
    """
    Load all YOLO IMAGE class file in the folder into a list and return it
    """

    images = []
    paths = []

    for i, file in enumerate([f for f in os.listdir(images_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]):
        paths.insert(i, images_folder + file)
        images.append(wrapper.load_image(bytes(paths[i], encoding='utf-8'), 0, 0))
        
    
    return images, paths

def in_front_of_both_cameras(first_points, second_points, rot, trans):
    # check if the point correspondences are in front of both images
    rot_inv = rot
    for first, second in zip(first_points, second_points):
        first_z = np.dot(rot[0, :] - second[0]*rot[2, :], trans) / np.dot(rot[0, :] - second[0]*rot[2, :], second)
        first_3d_point = np.array([first[0] * first_z, second[0] * first_z, first_z])
        second_3d_point = np.dot(rot.T, first_3d_point) - np.dot(rot.T, trans)

        if first_3d_point[2] < 0 or second_3d_point[2] < 0:
            return False

    return True

def retrieve_common_kps(matches):
    """
    Takes the DMatch-es of the fisrt photo with each one of the other and returns the coordinates of the first photo that are matched in each one of the other photos
    """
    if len(matches) == 1:
        return matches
    for i, _ in enumerate(matches):
        if i==0:
            temp = list(set(matches[i][0]).intersection(matches[i+1][0]))
        elif i < (len(matches)-1):
            temp = list(set(temp).intersection(matches[i+1][0]))
    
    return temp
        

# Monodepth2

def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines

def normalize_image(x):
    """Rescale image pixels to span range [0, 1]
    """
    ma = float(x.max().cpu().data)
    mi = float(x.min().cpu().data)
    d = ma - mi if ma != mi else 1e5
    return (x - mi) / d

def sec_to_hm(t):
    """Convert time in seconds to time in hours, minutes and seconds
    e.g. 10239 -> (2, 50, 39)
    """
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return t, m, s

def sec_to_hm_str(t):
    """Convert time in seconds to a nice string
    e.g. 10239 -> '02h50m39s'
    """
    h, m, s = sec_to_hm(t)
    return "{:02d}h{:02d}m{:02d}s".format(h, m, s)

def download_model_if_doesnt_exist(model_name):
    """If pretrained kitti model doesn't exist, download and unzip it
    """
    # values are tuples of (<google cloud URL>, <md5 checksum>)
    download_paths = {
        "mono_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_640x192.zip",
             "a964b8356e08a02d009609d9e3928f7c"),
        "stereo_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_640x192.zip",
             "3dfb76bcff0786e4ec07ac00f658dd07"),
        "mono+stereo_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_640x192.zip",
             "c024d69012485ed05d7eaa9617a96b81"),
        "mono_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_no_pt_640x192.zip",
             "9c2f071e35027c895a4728358ffc913a"),
        "stereo_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_no_pt_640x192.zip",
             "41ec2de112905f85541ac33a854742d1"),
        "mono+stereo_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_no_pt_640x192.zip",
             "46c3b824f541d143a45c37df65fbab0a"),
        "mono_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_1024x320.zip",
             "0ab0766efdfeea89a0d9ea8ba90e1e63"),
        "stereo_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_1024x320.zip",
             "afc2f2126d70cf3fdf26b550898b501a"),
        "mono+stereo_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_1024x320.zip",
             "cdc5fc9b23513c07d5b19235d9ef08f7"),
        }

    if not os.path.exists("models"):
        os.makedirs("models")

    model_path = os.path.join("models", model_name)

    def check_file_matches_md5(checksum, fpath):
        if not os.path.exists(fpath):
            return False
        with open(fpath, 'rb') as f:
            current_md5checksum = hashlib.md5(f.read()).hexdigest()
        return current_md5checksum == checksum

    # see if we have the model already downloaded...
    if not os.path.exists(os.path.join(model_path, "encoder.pth")):

        model_url, required_md5checksum = download_paths[model_name]

        if not check_file_matches_md5(required_md5checksum, model_path + ".zip"):
            print("-> Downloading pretrained model to {}".format(model_path + ".zip"))
            urllib.request.urlretrieve(model_url, model_path + ".zip")

        if not check_file_matches_md5(required_md5checksum, model_path + ".zip"):
            print("   Failed to download a file which matches the checksum - quitting")
            quit()

        print("   Unzipping model...")
        with zipfile.ZipFile(model_path + ".zip", 'r') as f:
            f.extractall(model_path)

        print("   Model unzipped to {}".format(model_path))

# SuperGlue

def retrieve_kps_superglue(pairs_folder):
    """
    Retrieve the keypoints from the output files of the SuperGlue network stored in @pairs_folder
    """
    alls = []
    for k,file in enumerate(sorted(glob.glob(pairs_folder + "*.npz"))):
        dict_matches = np.load(file)
        kps = []
        for i, kp in enumerate(list(dict_matches['keypoints0'])):
            if (dict_matches['matches'][i] > -1) and (dict_matches['match_confidence'][i] > .8):
                temp = (tuple(dict_matches['keypoints0'][i]), tuple(dict_matches['keypoints1'][dict_matches['matches'][i]]))
                #if temp[0] != temp[1]:
                kps.append((cv.KeyPoint(temp[0][0], temp[0][1], 0), cv.KeyPoint(temp[1][0], temp[1][1], 0)))
        alls.append(kps)
    h = []
    for i in alls[0]:
        h.append(i[0])

    l = []
    for i in alls[2]:
        l.append(i[1])

    return h
    #img = cv.imread("images/1.jpg", cv.IMREAD_GRAYSCALE)
    #last = cv.drawKeypoints(img, h,None)
    #plt.imshow(last),plt.show()
