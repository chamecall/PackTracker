# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

import PartDetection


class PartTracker:
    def __init__(self, maxDisappeared=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared

    def register(self, part_detection: PartDetection):
        self.objects[self.nextObjectID] = part_detection
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, input_part_detections):

        if len(input_part_detections) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            return self.objects.values()


        if len(self.objects) == 0:
            for part_detection in input_part_detections:
                self.register(part_detection)

        else:
            objectIDs = list(self.objects.keys())
            part_detection_centroids = [part_detection.object_shape.center for part_detection in self.objects.values()]
            input_part_detection_centroids = [part_detection.object_shape.center for part_detection in input_part_detections]
            D = dist.cdist(np.array(part_detection_centroids), np.array(input_part_detection_centroids))

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = input_part_detections[col]
                self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

            else:
                for col in unusedCols:
                    self.register(input_part_detections[col])

        return self.objects.values()
