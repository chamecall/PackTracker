from __future__ import print_function
import cv2 as cv2
import math

from ObjectShape import ObjectShape
import imutils
import numpy as np
from imutils import perspective
from PackBox import PackBox

class BoxDetector:
    def __init__(self):
        self.low_range_opened_box = (12, 86, 67)
        self.high_range_opened_box = (23, 154, 123)
        self.low_range_closed_box = (0, 20, 170)
        self.high_range_closed_box = (20, 115, 230)

    def detect_closed_boxes(self, frame):
        return self.detect_boxes(frame, self.low_range_closed_box, self.high_range_closed_box, PackBox.statuses['Closed'])

    def detect_opened_boxes(self, frame):
        return self.detect_boxes(frame, self.low_range_opened_box, self.high_range_opened_box, PackBox.statuses['Opened'])

    def detect_boxes(self, frame, low_range, high_range, status):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame_box_in = cv2.inRange(frame, low_range, high_range)
        frame_box_in = cv2.dilate(frame_box_in, (3, 3), iterations=1)
        contours, _ = cv2.findContours(frame_box_in, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        suited_box = None
        max_fullness = 65
        max_area_contour = 10000
        for contour in contours:
            contour_area = cv2.contourArea(contour)
            if contour_area > max_area_contour:
                pack_box = self.create_pack_box_from_contour(contour, status)
                if pack_box.fullness > max_fullness:
                    suited_box = pack_box
                    max_fullness = pack_box.fullness
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


    # def fitler_opened_boxes(self, frame, opened_boxes):
    #     for box in opened_boxes:
    #         box_rect = box.object_shape.box_rect
    #         print(box_rect)
    #         #print(box_rect)
    #         result_frame = frame[box_rect[1]:box_rect[1]+box_rect[3], box_rect[0]:box_rect[0]+box_rect[2]]
    #         # cv2.rectangle(frame, box_rect, (255, 0, 255), 2)
    #         # cv2.imshow('sdf', result_frame)
    #         boxes_in_the_rect = self.detect_opened_boxes(result_frame)
    #         if not boxes_in_the_rect:
    #             del opened_boxes[0]
    #             del
    #         else:
    #             opened_boxes[0] = boxes_in_the_rect[0]
    #             print('old box : ', box.fullness)
    #             print('new box : ', boxes_in_the_rect[0].fullness)
