import numpy as np
import cv2 as cv


def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

dist = np.array([-0.03432, 0.05332, -0.00347, 0.00106, 0.00000, 0.0, 0.0, 0.0]).reshape(1, 8) # distortion coefficients
mtx = np.array([1189.46, 0.0, 805.49, 0.0, 1191.78, 597.44, 0.0, 0.0, 1.0]).reshape(3, 3) # Camera matrix

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6*7,3), np.float32)

objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

img = cv.imread('images/left04.jpg')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
ret, corners = cv.findChessboardCorners(gray, (7,6),None)
print(corners.shape)
if ret == True:
    corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
    print(corners2.shape)
    # Find the rotation and translation vectors.
    ret,rvecs, tvecs, _ = cv.solvePnPRansac(objp, corners2, mtx, dist)
    # project 3D points to image plane
    imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
    print(imgpts.shape)
    img = draw(img,corners2,imgpts)
    cv.imshow('img',img)
    k = cv.waitKey(0) & 0xFF
    if k == ord('s'):
        cv.imwrite(fname[:6]+'.png', img)