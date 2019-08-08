import Part
from ObjectShape import ObjectShape


class PartDetection:
    statuses = {'Unknown': 0, 'OutOfBox': 1, 'InBox': 2}

    def __init__(self, part: Part, object_shape: ObjectShape, status=statuses['Unknown']):
        self.part = part
        self.object_shape = object_shape
        self.status = status
