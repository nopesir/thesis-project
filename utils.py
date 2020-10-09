import wrapper
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
    Filter out the keypoints not in the bbox and discards the ALL-n ones that are far from the bbox center
    """

    kp_yolo = []

    for keypoint in kp:
        if (keypoint.pt[0] >= bbox[0]) and (keypoint.pt[0] <= bbox[2]) and (keypoint.pt[1] >= bbox[1])  and (keypoint.pt[1] <= bbox[3]):
            kp_yolo.append(keypoint)
    


    kp_yolo.sort(key = lambda p: (p.pt[0] - kp_center.pt[0])**2 + (p.pt[1] - kp_center.pt[1])**2)


    kp_yolo = kp_yolo[:n]

    return kp_yolo



def apply_yolo_orb(img, bbox, kp_center):
    """
    Apply ORB on the bbox by filtering the keypoints and use L2 norm distance from the center
    """

    # Initiate ORB detector
    orb = cv.ORB_create()

    # find the keypoints with ORB
    kp = orb.detect(img, None)

    # compute the descriptors with ORB
    kp, _ = orb.compute(img, kp)

    kp_yolo = kp_filtersort_L2(kp, bbox, kp_center)

    return kp_yolo
    
