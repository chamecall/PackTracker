from ObjectShape import ObjectShape


class PackBox:
    statuses = {'Opened': 1, 'Closed': 0}

    def __init__(self, object_shape: ObjectShape, status):
        self.shape = object_shape
        self.status = status
