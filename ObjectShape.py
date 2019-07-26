from scipy.spatial import distance as dist
from Utils import midpoint


class ObjectShape:
    def __init__(self, points: tuple, center: tuple):
        self.tl_point = points[0]
        self.tr_point = points[1]
        self.br_point = points[2]
        self.bl_point = points[3]
        self.points = [self.tl_point, self.tr_point, self.br_point, self.bl_point]

        self.lm_point = midpoint(self.tl_point, self.bl_point)
        self.tm_point = midpoint(self.tl_point, self.tr_point)
        self.rm_point = midpoint(self.tr_point, self.br_point)
        self.bm_point = midpoint(self.bl_point, self.br_point)
        self.center = center

        self.height = int(dist.euclidean(self.tm_point, self.bm_point))
        self.width = int(dist.euclidean(self.lm_point, self.rm_point))



