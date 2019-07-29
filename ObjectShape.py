from scipy.spatial import distance as dist
from Utils import midpoint


class ObjectShape:
    def __init__(self, points: tuple):
        self.tl_point = points[0]
        self.tr_point = points[1]
        self.br_point = points[2]
        self.bl_point = points[3]
        self.points = [self.tl_point, self.tr_point, self.br_point, self.bl_point]

        xs = [point[0] for point in self.points]
        min_x, max_x = min(xs), max(xs)
        ys = [point[1] for point in self.points]
        min_y, max_y = min(ys), max(ys)

        self.center = int((min_x + max_x) / 2), int((min_y + max_y) / 2)

        self.lm_point = midpoint(self.tl_point, self.bl_point)
        self.tm_point = midpoint(self.tl_point, self.tr_point)
        self.rm_point = midpoint(self.tr_point, self.br_point)
        self.bm_point = midpoint(self.bl_point, self.br_point)

        self.height = int(dist.euclidean(self.tm_point, self.bm_point))
        self.width = int(dist.euclidean(self.lm_point, self.rm_point))
