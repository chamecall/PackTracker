from scipy.spatial import distance as dist
from Utils import midpoint


class ObjectShape:
    def __init__(self, rect: tuple):
        rect = tuple(int(num) for num in rect)
        self.box_rect = rect
        self.box_rect_center = tuple([int(rect[i] + rect[i + 2] / 2) for i in range(2)])
        self.tm_point_rect = int(rect[0] + rect[2] / 2), rect[1]
        self.rm_point_rect = rect[0] + rect[2], int(rect[1] + rect[3] / 2)

