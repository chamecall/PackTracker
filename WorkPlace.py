def stretch_place_to_left(place):
    place[0][0] -= WorkPlace.WORKER_PLACE_SIZE
    return place


def stretch_place_to_right(place):
    place[1][0] += WorkPlace.WORKER_PLACE_SIZE
    return place


def stretch_place_to_top(place):
    place[0][1] -= WorkPlace.WORKER_PLACE_SIZE
    return place


def stretch_place_to_bottom(place):
    place[1][1] += WorkPlace.WORKER_PLACE_SIZE
    return place


calculate_work_size = {
    'Left': stretch_place_to_left,
    'Right': stretch_place_to_right,
    'Top': stretch_place_to_top,
    'Bottom': stretch_place_to_bottom
}


class WorkPlace:
    WORKER_PLACE_SIZE = 300

    def __init__(self, packer, table_position: tuple, packing_side, clockwise_distance_coeffs, frame_size=(1366, 768)):
        self.packer = packer
        self.table_position = table_position
        self.packing_side = packing_side
        self.frame_size = frame_size
        self.work_place_position = self.define_work_place_size()
        self.distance_coeffs = {'tl': clockwise_distance_coeffs[0], 'tr': clockwise_distance_coeffs[1],
                                'br': clockwise_distance_coeffs[2], 'bl': clockwise_distance_coeffs[3]}

    def define_work_place_size(self):
        table_position_copy = [list(tpl) for tpl in self.table_position]
        work_place_position = calculate_work_size[self.packing_side](table_position_copy)
        work_place_position[0][0] = 0 if work_place_position[0][0] < 0 else work_place_position[0][0]
        work_place_position[0][1] = 0 if work_place_position[0][1] < 0 else work_place_position[0][1]
        work_place_position[1][0] = self.frame_size[0] if work_place_position[1][0] > self.frame_size[0] else \
        work_place_position[1][0]
        work_place_position[1][1] = self.frame_size[1] if work_place_position[1][1] > self.frame_size[1] else \
        work_place_position[1][1]
        return tuple(tuple(lst) for lst in work_place_position)

