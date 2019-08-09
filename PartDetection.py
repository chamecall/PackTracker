import Part
from ObjectShape import ObjectShape


class PartDetection:
    statuses = {'Tracked': 0, 'Moved': 1, 'InBox': 2}

    def __init__(self, part: Part, object_shape: ObjectShape, status=statuses['Tracked']):
        self.part = part
        self.object_shape = object_shape
        self.status = status

    def is_tracked(self):
        return self.status == PartDetection.statuses['Tracked']

    def is_moved(self):
        return self.status == PartDetection.statuses['Moved']

    def is_in_box(self):
        return self.status == PartDetection.statuses['InBox']

    def set_status_as_tracked(self):
        self.status = PartDetection.statuses['Tracked']

    def set_status_as_moved(self):
        self.status = PartDetection.statuses['Moved']

    def set_status_as_in_box(self):
        self.status = PartDetection.statuses['InBox']
