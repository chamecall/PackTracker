import json
from os import walk
import os

class JsonReader:
    def __init__(self, dir_name):
        self.json_dir = dir_name
        self.json_files = self.get_json_files_from_dir(dir_name)
        self.json_file_num = -1

        self.required_points = {"Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4, "LShoulder": 5,
                                "LElbow": 6, "LWrist": 7}

        self.required_pairs = (
            ['Neck', 'RShoulder'], ['Neck', 'LShoulder'], ['RShoulder', 'RElbow'], ['LShoulder', 'LElbow'],
            ['RElbow', 'RWrist'], ['LElbow', 'LWrist'],
            ['Neck', 'MidHip'],
            ['MidHip', 'LHip'], ['MidHip', 'RHip'], ['RHip', 'RKnee'], ['LHip', 'LKnee'], ['LKnee', 'LAnkle'],
            ['RKnee', 'RAnkle'])

    def read(self):
        self.json_file_num += 1
        filename = self.json_files[self.json_file_num]
        full_filename_path = os.path.join(self.json_dir, filename)
        points = []
        with open(full_filename_path, "r") as json_data:
            data = json.load(json_data)

            if data['people']:
                points_list = data['people'][0]['pose_keypoints_2d']
                points = self.extract_required_json_points(points_list, self.required_points)

        return points

    @staticmethod
    def get_json_files_from_dir(json_dir):
        json_files = []
        for _, _, f_names in walk(json_dir):
            json_files.extend(f_names)
            break
        json_files = sorted(json_files)
        return json_files

    def extract_required_json_points(self, points_list, needed_points: dict):
        points = {}
        for joint, joint_number in needed_points.items():
            position = 3 * joint_number
            x, y = (int(points_list[position]), int(points_list[position + 1]))
            if 0 < x < 10 and 0 < y < 10:
                print(x, y)
            if (x, y) == (0, 0):
                points[joint] = None
            else:

                points[joint] = (x, y)
        return points

reader = JsonReader('/home/algernon/PycharmProjects/test/origin_json')
print(reader.read())
