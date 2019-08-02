from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
from ObjectShape import ObjectShape


def find_contours(image, alignment_point=(0, 0)):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    edged = cv2.Canny(gray, 30, 70)

    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    # cv2.imshow("Frame", edged)
    # cv2.waitKey(1)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if cnts:
        cnts, _ = contours.sort_contours(cnts)

    corners = []
    for c in cnts:
        if not 500 < cv2.contourArea(c) < 3000:
            continue

        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = [[point[0] + alignment_point[0], point[1] + alignment_point[1]] for point in box]
        box = np.array(box, dtype="int")

        box = perspective.order_points(box)

        (tl, tr, br, bl) = box

        corners.append(ObjectShape((tl, tr, br, bl)))
    return corners


def apply_info(object_shape: ObjectShape, image):
    cv2.circle(image, object_shape.tm_point, 5, (255, 0, 0), -1)
    cv2.circle(image, object_shape.bm_point, 5, (255, 0, 0), -1)
    cv2.circle(image, object_shape.lm_point, 5, (255, 0, 0), -1)
    cv2.circle(image, object_shape.rm_point, 5, (255, 0, 0), -1)

    cv2.line(image, object_shape.tm_point, object_shape.bm_point,
             (255, 0, 255), 2)
    cv2.line(image, object_shape.lm_point, object_shape.rm_point,
             (255, 0, 255), 2)
