from __future__ import print_function
import cv2 as cv2
import math

from ObjectShape import ObjectShape
import imutils
import numpy as np
from imutils import perspective
from PackBox import PackBox

class BoxDetector:

    def detect_closed_boxes(self, frame):
        return self.detect_boxes(frame, self.low_range_closed_box, self.high_range_closed_box, PackBox.statuses['Closed'])

    def detect_opened_boxes(self, table_rect, box_detections):
        return self.detect_boxes(table_rect, box_detections, PackBox.statuses['Opened'])

    def detect_boxes(self, table_rect, box_detections, status):

        return [suited_box] if suited_box else []

    def create_pack_box_from_contour(self, contour, status):

        box = cv2.minAreaRect(contour)
        box = cv2.boxPoints(box)

        box = np.array(box, dtype="int")
        box = perspective.order_points(box)
        box = np.array(box, dtype="int")
        (tl, tr, br, bl) = box
        object_shape = ObjectShape((tl, tr, br, bl))
        # print('area', cv2.contourArea(contour))
        # print('width', object_shape.width)
        # print('height', object_shape.height)
        fullness = int(cv2.contourArea(contour) / (object_shape.width * object_shape.height) * 100)
        print()
        box = PackBox(object_shape, status, cv2.contourArea(contour), fullness)

        return box


