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
        self.rect_box = None
        self.rect_box_center = None
        self.rect_tm_point = None
        self.rect_rm_point = None

        self.height = int(dist.euclidean(self.tl_point, self.bl_point))
        self.width = int(dist.euclidean(self.tl_point, self.tr_point))
        self.set_rect_box((*rect_tl_point, rect_width, rect_height))

    def set_rect_box(self, new_rect_box):
        self.rect_box = new_rect_box
        self.rect_box_center = tuple([int(new_rect_box[i] + new_rect_box[i + 2] / 2) for i in range(2)])
        rect_tl_point = (new_rect_box[0], new_rect_box[1])
        rect_width = new_rect_box[2]
        rect_height = new_rect_box[3]
        self.rect_tm_point = (int(rect_tl_point[0] + rect_width / 2), rect_tl_point[1])
        self.rect_rm_point = (rect_tl_point[0] + rect_width, int(rect_tl_point[1] + rect_height / 2))
