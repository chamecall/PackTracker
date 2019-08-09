import json
from os import walk
import os
import cv2

class JsonReader:
    def __init__(self, dir_name):
        self.json_dir = dir_name
        self.json_files = self.get_json_files_from_dir(dir_name)
        self.json_file_num = -1

        self.required_points = {"Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4, "LShoulder": 5,
                                "LElbow": 6, "LWrist": 7}

        self.required_pairs = (
            ['Neck', 'RShoulder'], ['Neck', 'LShoulder'], ['RShoulder', 'RElbow'], ['LShoulder', 'LElbow'],
            ['RElbow', 'RWrist'], ['LElbow', 'LWrist'])

    def read(self, frame):
        self.json_file_num += 1
        filename = self.json_files[self.json_file_num]
        full_filename_path = os.path.join(self.json_dir, filename)
        points = {}
        with open(full_filename_path, "r") as json_data:
            data = json.load(json_data)

            if data['people']:
                points_list = data['people'][0]['pose_keypoints_2d']
                points = self.extract_required_json_points(points_list, self.required_points)
        self.draw_skeleton(frame, points)

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

    def draw_skeleton(self, frame, points, line_color=(0, 0, 255), circle_color=(0, 0, 255), radius=8,
                      line_thickness=3):

        for pair in self.required_pairs:
            part_a = pair[0]
            part_b = pair[1]

            if points.get(part_a) and points.get(part_b):
                cv2.line(frame, points[part_a], points[part_b], line_color, line_thickness, lineType=cv2.LINE_AA)
                draw_point(frame, points[part_a], circle_color, radius)
                draw_point(frame, points[part_b], circle_color, radius)



def draw_point(frame, point, color, radius):
    cv2.circle(frame, (point[0], point[1]), radius, color, thickness=-1, lineType=cv2.FILLED)

# reader = JsonReader('/home/algernon/PycharmProjects/test/origin_json')
# print(reader.read())
