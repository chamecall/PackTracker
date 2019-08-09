import cv2


class Hands:
    required_pairs = (
        ['Neck', 'RShoulder'], ['Neck', 'LShoulder'], ['RShoulder', 'RElbow'], ['LShoulder', 'LElbow'],
        ['RElbow', 'RWrist'], ['LElbow', 'LWrist'])

    def __init__(self):
        self.l_wrist_point = ()
        self.r_wrist_point = ()
        self.points = {}

    def set_new_points(self, points):
        if points:
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


def draw_point(frame, point, color, radius):
    cv2.circle(frame, (point[0], point[1]), radius, color, thickness=-1, lineType=cv2.FILLED)
