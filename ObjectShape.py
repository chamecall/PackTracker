from scipy.spatial import distance as dist


class ObjectShape:
    def __init__(self, m_points: tuple, center: tuple):
        self.lm_point = m_points[0]
        self.tm_point = m_points[1]
        self.rm_point = m_points[2]
        self.bm_point = m_points[3]
        self.center = center

        self.height = int(dist.euclidean(self.tm_point, self.bm_point))
        self.width = int(dist.euclidean(self.lm_point, self.rm_point))
