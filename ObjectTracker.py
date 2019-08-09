# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import cv2
from PartDetection import PartDetection
from ObjectShape import ObjectShape
from scipy.spatial import distance as dist
from Part import Part
from collections import deque

class ObjectTracker:
    def __init__(self, distance_threshold=50):
        self.trackers = []
        self.detections = []
        self.distance_threshold = distance_threshold
        self.shift_histories = []

    def get_detections(self):
        return self.detections

    def update(self, frame):
        for i, tracker in enumerate(self.trackers):
            success, box = tracker.update(frame)
            if success:
                old_center = self.detections[i].object_shape.box_rect_center
                self.detections[i].object_shape.set_box_rect(tuple(int(v if v >= 0 else 0) for v in box))
                new_center = self.detections[i].object_shape.box_rect_center
                shift = dist.cdist([new_center], [old_center])[0][0]

                self.shift_histories[i].append(shift)
            else:
                del self.trackers[i]
                del self.detections[i]
                del self.shift_histories[i]
        return self.detections

    def track(self, frame, new_detections):
        self.update(frame)
        self.init(frame, new_detections)
        return self.detections

    def del_tracker(self, part):
        index = [detection.part for detection in self.detections].index(part)
        del self.trackers[index]
        del self.detections[index]
        del self.shift_histories[index]

    def get_shift_histories_by_part(self, part: Part):
        index = [detection.part for detection in self.detections].index(part)
        return self.shift_histories[index]

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

            tracker = cv2.TrackerCSRT_create()
            tracker.init(frame, new_box)
            self.trackers.append(tracker)
            self.detections.append(new_detection)
            deq = deque(maxlen=5)
            deq.append(0)
            self.shift_histories.append(deq)


