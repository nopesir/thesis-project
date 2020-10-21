import numpy as np
import cv2 as cv
import glob
from config import *


#def draw(img, corners, imgpts):
#    corner = tuple(corners[0].ravel())
#    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
#    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
#    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
#    return img

def draw(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    # draw ground floor in green
    img = cv.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)
    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
    # draw top layer in red color
    img = cv.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)
    return img

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
#axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
axis = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
                   [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])

for fname in glob.glob('images/*.jpg'):
    img = cv.imread(fname)
    img = cv.resize(img,None,fx=0.4,fy=0.4)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (7,6),None)
    if ret == True:
        corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        # Find the rotation and translation vectors.
        ret,rvecs, tvecs, _ = cv.solvePnPRansac(objp, corners2, K, d)
        # project 3D points to image plane
        imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, K, d)
        img = draw(img,corners2,imgpts)
        cv.imshow('img',img)
        k = cv.waitKey(1000)
        if k == ord('s'):
            cv.imwrite(fname[:6]+'.png', img)
cv.destroyAllWindows()