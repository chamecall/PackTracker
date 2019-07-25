from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

from ObjectShape import ObjectShape
from WorkPlace import WorkPlace
import Utils
from collections import deque


def midpoint(ptA, ptB):
    return int((ptA[0] + ptB[0]) * 0.5), int((ptA[1] + ptB[1]) * 0.5)


def get_clockwise_midside_points(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (13, 13), 0)

    edged = cv2.Canny(gray, 30, 70)

    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    # cv2.imshow("Frame", edged)
    # cv2.waitKey(1)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    (cnts, _) = contours.sort_contours(cnts)

    corners = []
    for c in cnts:
        if not 500 < cv2.contourArea(c) < 3000:
            continue

        image = image
        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")

        box = perspective.order_points(box)
        cv2.drawContours(image, [box.astype("int")], -1, (0, 255, 0), 2)

        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        for (x, y) in box:
            cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)

        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)

        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        corners.append(ObjectShape(((tlblX, tlblY), (tltrX, tltrY), (trbrX, trbrY), (blbrX, blbrY)), (cX, cY)))
    return corners


# def process_video(video_name):
#     cap = cv2.VideoCapture(video_name)
#
#     cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     vid_writer = cv2.VideoWriter('edge_output.avi', cv2.VideoWriter_fourcc(*"XVID"), fps/3,
#                                  (cap_width, cap_height))
#
#
#     cv2.namedWindow('Frame', cv2.WND_PROP_FULLSCREEN)
#     cv2.setWindowProperty("Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
#
#     while True:
#         captured, image = cap.read()
#         if not captured:
#             break
#
#         rectangles = deque(get_clockwise_corner_points(image))
#
#
#
#         for rectangle in rectangles:
#             apply_info(rectangle, image)
#
#         cv2.rectangle(image, bounding_areas[0][0], bounding_areas[0][1], (0, 0, 255), 5)
#         cv2.imshow("Frame", image)
#         vid_writer.write(image)
#         cv2.waitKey(1)
#     vid_writer.release()


def apply_info(object_shape: ObjectShape, image, work_place):

    cv2.circle(image, object_shape.tm_point, 5, (255, 0, 0), -1)
    cv2.circle(image, object_shape.bm_point, 5, (255, 0, 0), -1)
    cv2.circle(image, object_shape.lm_point, 5, (255, 0, 0), -1)
    cv2.circle(image, object_shape.rm_point, 5, (255, 0, 0), -1)

    distance_coeff = work_place.calculate_distance_coeff_by_point(object_shape.center)

    cv2.line(image, object_shape.tm_point, object_shape.bm_point,
             (255, 0, 255), 2)
    cv2.line(image, object_shape.lm_point, object_shape.rm_point,
             (255, 0, 255), 2)

    dimA = object_shape.width * distance_coeff
    dimB = object_shape.height * distance_coeff

    cv2.putText(image, "{:.1f}mm".format(dimA),
                (object_shape.tm_point[0] - 15, object_shape.tm_point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2)
    cv2.putText(image, "{:.1f}mm".format(dimB),
                (object_shape.rm_point[0] + 10, object_shape.rm_point[1]), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2)

# if __name__ == '__main__':
#     process_video(
#         '/home/algernon/samba/video_queue/omega-packaging/data/raw/МЕЛ 1 стол упаковки 1 _20190701-124357--20190701-144357_EDIT.avi')
