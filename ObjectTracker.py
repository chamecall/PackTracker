# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import cv2
from PartDetection import PartDetection
from ObjectShape import ObjectShape
from scipy.spatial import distance as dist
from Part import Part


class ObjectTracker:
    def __init__(self, distance_threshold=50):
        self.trackers = []
        self.detections = []
        self.distance_threshold = distance_threshold

    def get_detections(self):
        return self.detections

    def update(self, frame):
        for i, tracker in enumerate(self.trackers):
            success, box = tracker.update(frame)
            if success:
                self.detections[i].object_shape.box_rect = tuple(int(v if v >= 0 else 0) for v in box)
            else:
                del self.trackers[i]
                del self.detections[i]

    def track(self, frame, new_detections):
        self.update(frame)
        self.init(frame, new_detections)

    def init(self, frame, new_detections):
        for new_detection in new_detections:
            new_box = new_detection.object_shape.box_rect
            if self.detections:
                new_box_center = new_detection.object_shape.box_rect_center
                detected_box_centers = (part_detection.object_shape.box_rect_center for part_detection in
                                        self.detections)
                is_box_valid = all([int(dist.euclidean(new_box_center, old_box_center)) > self.distance_threshold \
                                    for old_box_center in detected_box_centers])
                if not is_box_valid:
                    return

            tracker = cv2.TrackerMedianFlow_create()
            print(new_box)
            tracker.init(frame, new_box)
            self.trackers.append(tracker)
            self.detections.append(new_detection)
