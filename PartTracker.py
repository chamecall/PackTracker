# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import cv2
import PartDetection
from scipy.spatial import distance as dist


class PartTracker:
    def __init__(self, distance_threshold=10):
        self.trackers = cv2.MultiTracker_create()
        self.part_detections = []
        self.distance_threshold = distance_threshold

    def get_part_detections(self):
        return self.part_detections

    def update(self, frame):
        success, boxes = self.trackers.update(frame)
        for i, box in enumerate(boxes):
            self.part_detections[i].object_shape.rect_box = [int(v) for v in box]

    def init(self, frame, new_part_detections):
        self.update(frame)

        for new_part_detection in new_part_detections:
            new_box = new_part_detection.object_shape.rect_box
            if self.part_detections:
                new_box_center = new_part_detection.object_shape.rect_box_center
                detected_box_centers = (part_detection.object_shape.rect_box_center for part_detection in self.part_detections)
                is_box_valid = all([int(dist.euclidean(new_box_center, old_box_center)) > self.distance_threshold \
                                    for old_box_center in detected_box_centers])
                if not is_box_valid:
                    continue



            tracker = cv2.TrackerCSRT_create()
            self.trackers.add(tracker, frame, new_box)
            self.part_detections.append(new_part_detection)
