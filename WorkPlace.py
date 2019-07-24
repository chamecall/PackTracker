import copy
import math


def stretch_place_to_left(rect_table_corners):
    rect_table_corners['tl'][0] -= WorkPlace.WORKER_PLACE_SIZE
    return rect_table_corners


def stretch_place_to_right(rect_table_corners):
    rect_table_corners['br'][0] += WorkPlace.WORKER_PLACE_SIZE
    return rect_table_corners


def stretch_place_to_top(rect_table_corners):
    rect_table_corners['tl'][1] -= WorkPlace.WORKER_PLACE_SIZE
    return rect_table_corners


def stretch_place_to_bottom(rect_table_corners):
    rect_table_corners['br'][1] += WorkPlace.WORKER_PLACE_SIZE
    return rect_table_corners


calculate_work_place_size = {
    'Left': stretch_place_to_left,
    'Right': stretch_place_to_right,
    'Top': stretch_place_to_top,
    'Bottom': stretch_place_to_bottom
}


class WorkPlace:
    WORKER_PLACE_SIZE = 300
    def calculate_rect_table_area(self):
        tcs = self.table_corners
        min_x, min_y = min(tcs['tl'][0], tcs['bl'][0]), min(tcs['tl'][1], tcs['tr'][1])
        max_x, max_y = max(tcs['tr'][0], tcs['br'][0]), max(tcs['bl'][1], tcs['br'][1])
        rect_table_corners = {'tl': (min_x, min_y),
                              'br': (max_x, max_y)}
        return rect_table_corners


    def __init__(self, packer, table_corners: tuple, packing_side, table_corners_distance_coeffs, frame_size=(1366, 768)):
        self.packer = packer
        self.table_corners = {'tl': table_corners[0], 'tr': table_corners[1],
                                'br': table_corners[2], 'bl': table_corners[3]}

        self.packing_side = packing_side
        self.frame_size = frame_size

        self.table_corner_distance_coeffs = {'tl': table_corners_distance_coeffs[0], 'tr': table_corners_distance_coeffs[1],
                                'br': table_corners_distance_coeffs[2], 'bl': table_corners_distance_coeffs[3]}
        self.rect_table_corners = self.calculate_rect_table_area()

        self.rect_table_corner_distance_coeffs = self.calculate_rect_table_corner_distance_coeffs()

        self.rect_work_place_corners = self.define_work_place_corners()
        self.width = self.rect_table_corners['br'][0] - self.rect_table_corners['tl'][0]
        self.height = self.rect_table_corners['br'][1] - self.rect_table_corners['tl'][1]

        self.distance_coeff_width = abs(self.rect_table_corner_distance_coeffs['tr'] - self.rect_table_corner_distance_coeffs['tl'])
        self.distance_coeff_height = abs(
            self.rect_table_corner_distance_coeffs['bl'] - self.rect_table_corner_distance_coeffs['br'])

    def calculate_rect_table_corner_distance_coeffs(self):
        tb, rect_tb = self.table_corners, self.rect_table_corners
        tb_coeffs, rect_tb_coeffs = self.table_corner_distance_coeffs, {}
        top_coeff_per_pix = ((tb_coeffs['tr'] - tb_coeffs['tl']) / (tb['tr'][0] - tb['tl'][0]))
        bottom_coeff_per_pix = ((tb_coeffs['br'] - tb_coeffs['bl']) / (tb['br'][0] - tb['bl'][0]))

        rect_tb_coeffs['tl'] = tb_coeffs['tl'] + (rect_tb['tl'][0] - tb['tl'][0]) * top_coeff_per_pix
        rect_tb_coeffs['tr'] = tb_coeffs['tr'] + (rect_tb['br'][0] - tb['tr'][0]) * top_coeff_per_pix
        rect_tb_coeffs['bl'] = tb_coeffs['bl'] + (rect_tb['tl'][0] - tb['bl'][0]) * bottom_coeff_per_pix
        rect_tb_coeffs['br'] = tb_coeffs['br'] + (rect_tb['br'][0] - tb['br'][0]) * bottom_coeff_per_pix

        return rect_tb_coeffs

    def define_work_place_corners(self):
        rect_table_corners_copy = copy.deepcopy(self.rect_table_corners)
        rect_table_corners_copy = {key: list(value) for key, value in rect_table_corners_copy.items()}
        rect_work_place_corners = calculate_work_place_size[self.packing_side](rect_table_corners_copy)

        rect_work_place_corners['tl'][0] = 0 if rect_work_place_corners['tl'][0] < 0 else rect_work_place_corners['tl'][0]
        rect_work_place_corners['tl'][1] = 0 if rect_work_place_corners['tl'][1] < 0 else rect_work_place_corners['tl'][1]
        rect_work_place_corners['br'][0] = self.frame_size[0] if rect_work_place_corners['br'][0] > self.frame_size[0] else \
        rect_work_place_corners['br'][0]
        rect_work_place_corners['br'][1] = self.frame_size[1] if rect_work_place_corners['br'][1] > self.frame_size[1] else \
        rect_work_place_corners['br'][1]
        return rect_work_place_corners

    def calculate_distance_coeff_by_point(self, point):
        point_x_ratio = point[0] / self.width
        point_y_ratio = (self.width - point[1]) / self.height

        distance_coeff_x = self.distance_coeff_width * point_x_ratio
        distance_coeff_y = self.distance_coeff_height * point_y_ratio
        point_distance_coeff = math.sqrt(distance_coeff_x ** 2 + distance_coeff_y ** 2)
        point_distance_coeff += self.rect_table_corner_distance_coeffs['bl']
        return point_distance_coeff




