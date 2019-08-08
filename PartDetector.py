from Utils import is_num_in_range, calculate_two_sizes_match_precision
from PartDetection import PartDetection
import cv2
import numpy as np
from ObjectShape import ObjectShape

class PartDetector:
    def __init__(self, ):
        pass

    def detect(self, pack_tasks, parts_detections):
        best_precision_detections = []

        not_completed_tasks = [pack_task for pack_task in pack_tasks if not pack_task.is_detected()]

        for not_completed_task in not_completed_tasks:
            the_part_detections = [part_detection for part_detection in parts_detections if part_detection[0] == not_completed_task.part_class]
            if the_part_detections:
                best_part_detection = max(the_part_detections, key=lambda the_part_detection: the_part_detection[1])
                best_precision_detections.append(PartDetection(not_completed_task.part, ObjectShape(best_part_detection[2])))
        return best_precision_detections

