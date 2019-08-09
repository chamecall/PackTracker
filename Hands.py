import cv2
from scipy.spatial import distance
import math

class Hands:
    required_pairs = (
        ['Neck', 'RShoulder'], ['Neck', 'LShoulder'], ['RShoulder', 'RElbow'], ['LShoulder', 'LElbow'],
        ['RElbow', 'RWrist'], ['LElbow', 'LWrist'])

    def __init__(self, tl_bounding_point):
        self.l_wrist_point = ()
        self.r_wrist_point = ()
        self.points = {}
        self.tl_bounding_point = tl_bounding_point

    def set_new_points(self, points):
        if points:
            points = {part: (point[0] - self.tl_bounding_point[0], point[1] - self.tl_bounding_point[1]) if point else None for part, point in points.items()}
            self.points = points
            self.l_wrist_point = self.points['LWrist']
            self.r_wrist_point = self.points['RWrist']

    def draw_skeleton(self, frame, line_color=(0, 0, 255), circle_color=(0, 0, 255), radius=8,
                      line_thickness=3):

        for pair in self.required_pairs:
            part_a = pair[0]
            part_b = pair[1]

            if self.points.get(part_a) and self.points.get(part_b):
                cv2.line(frame, self.points[part_a], self.points[part_b], line_color, line_thickness, lineType=cv2.LINE_AA)
                draw_point(frame, self.points[part_a], circle_color, radius)
                draw_point(frame, self.points[part_b], circle_color, radius)

    def get_nearest_wrist_to_rect(self, rect):
        points = [(rect[0], rect[1]), (rect[0] + rect[2], rect[1]), (rect[0], rect[1] + rect[3]), (rect[0] + rect[2], rect[1] + rect[3])]
        nearest_dist = math.inf
        near_wrist = None
        if not self.l_wrist_point and not self.r_wrist_point:
            return None, math.inf

        for point in points:
            from_point_to_l_wrist = distance.cdist([point], [self.l_wrist_point]) if self.l_wrist_point else math.inf
            from_point_to_r_wrist = distance.cdist([point], [self.r_wrist_point]) if self.r_wrist_point else math.inf

            if from_point_to_l_wrist <= from_point_to_r_wrist:
                if from_point_to_l_wrist < nearest_dist:
                    nearest_dist = from_point_to_l_wrist
                    near_wrist = 'LWrist'
            else:
                if from_point_to_r_wrist < nearest_dist:
                    nearest_dist = from_point_to_r_wrist
                    near_wrist = 'RWrist'

        return near_wrist, nearest_dist


def draw_point(frame, point, color, radius):
    cv2.circle(frame, (point[0], point[1]), radius, color, thickness=-1, lineType=cv2.FILLED)
