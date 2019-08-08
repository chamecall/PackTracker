import json

class Detector:
    def __init__(self, json_file):
        json_data = open(json_file)
        self.detections = json.load(json_data)['boxes_detections']
        self.frame_num = -1


    def get_detections_per_frame(self):
        self.frame_num += 1
        return self.detections[self.frame_num]

