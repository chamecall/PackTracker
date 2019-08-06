from scipy.spatial import distance as dist
from Utils import midpoint


class ObjectShape:
    def __init__(self, points: tuple):

        self.tl_point = points[0]
        self.tr_point = points[1]
        self.br_point = points[2]
        self.bl_point = points[3]
        points = [self.tl_point, self.tr_point, self.br_point, self.bl_point]
        xs = [int(point[0]) for point in points]
        min_x, max_x = min(xs), max(xs)
        ys = [int(point[1]) for point in points]
        min_y, max_y = min(ys), max(ys)

        rect_tl_point = (min_x, min_y)
        rect_width, rect_height = max_x - min_x, max_y - min_y
        self.box_rect = None
        self.box_rect_center = None
        self.tm_point_rect = None
        self.rm_point_rect = None

        self.height = int(dist.euclidean(self.tl_point, self.bl_point))
        self.width = int(dist.euclidean(self.tl_point, self.tr_point))
        self.set_box_rect((*rect_tl_point, rect_width, rect_height))

    def set_box_rect(self, new_rect_box):
        self.box_rect = new_rect_box
        self.box_rect_center = tuple([int(new_rect_box[i] + new_rect_box[i + 2] / 2) for i in range(2)])
        tl_point_rect = (new_rect_box[0], new_rect_box[1])
        rect_width = new_rect_box[2]
        rect_height = new_rect_box[3]
        self.tm_point_rect = (int(tl_point_rect[0] + rect_width / 2), tl_point_rect[1])
        self.rm_point_rect = (tl_point_rect[0] + rect_width, int(tl_point_rect[1] + rect_height / 2))
