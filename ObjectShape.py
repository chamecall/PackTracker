from scipy.spatial import distance as dist
from Utils import midpoint


class ObjectShape:
    def __init__(self, rect: tuple):
        rect = tuple(int(num) for num in rect)
        self.box_rect = int(rect[0] - rect[2] / 2), int(rect[1] - rect[3] / 2), *rect[2:]
        self.box_rect_center = rect[0], rect[1]
        self.tm_point_rect = rect[0], int(rect[1] - rect[3] / 2)
        self.rm_point_rect = int(rect[0] + rect[2] / 2), rect[1]


    def set_box_rect(self, rect):
        self.box_rect = rect
        self.box_rect_center = int(rect[0] + rect[2] / 2), int(rect[1] + rect[3] / 2)
        self.tm_point_rect = int(rect[0] + rect[2] / 2), rect[1]
        self.rm_point_rect = rect[0] + rect[2], int(rect[1] + rect[3] / 2)

