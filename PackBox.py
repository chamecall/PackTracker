from ObjectShape import ObjectShape


class PackBox:
    statuses = {'Opened': 1, 'Closed': 0}
    reversed_statuses = {value: key for key, value in statuses.items()}

    def __init__(self, object_shape: ObjectShape, status):
        self.object_shape = object_shape
        self.status = status
