from __future__ import print_function
import cv2 as cv2
from ObjectShape import ObjectShape
import imutils
import numpy as np
from imutils import perspective
from PackBox import PackBox

class BoxDetector:
    def __init__(self):
        self.closed_boxes = []
        self.opened_boxes = []

    def detect_boxes(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame = cv2.GaussianBlur(frame, (7, 7), 0)
        frame_box_out = cv2.inRange(frame, (0, 20, 150), (20, 115, 230))
        frame_box_in = cv2.inRange(frame, (12, 86, 67), (23, 154, 123))
        frame_box_out = cv2.dilate(frame_box_out, (3, 3), iterations=1)
        frame_box_in = cv2.dilate(frame_box_in, (3, 3), iterations=1)

        # frame = cv2.bitwise_and(frame, frame, mask=cv2.add(frame_box_in, frame_box_out))

        contours, _ = cv2.findContours(frame_box_out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.closed_boxes = self._filter_box_contours(contours, PackBox.statuses['Closed'])
        contours, _ = cv2.findContours(frame_box_in, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.opened_boxes = self._filter_box_contours(contours, PackBox.statuses['Opened'])

    def _filter_box_contours(self, contours, status):
        box_areas = []
        for c in contours:
            if cv2.contourArea(c) < 5000:
                continue

            box = cv2.minAreaRect(c)
            box = cv2.boxPoints(box)

            box = np.array(box, dtype="int")
            box = perspective.order_points(box)
            box = np.array(box, dtype="int")
            (tl, tr, br, bl) = box
            object_shape = ObjectShape((tl, tr, br, bl))

            box_areas.append(PackBox(object_shape, status))

        return box_areas
