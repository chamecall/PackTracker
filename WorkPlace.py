import copy
import math
import cv2
import PackTask
from PartDetection import PartDetection
from Utils import is_num_in_range
from scipy.spatial import distance as dist


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

    def __init__(self, packer, table_corners: tuple, packing_side, table_corners_distance_coeffs,
                 frame_size=(1366, 768)):
        self.part_detections = []
        self.next_pack_tasks = None
        self.packer = packer
        self.table_corners = {'tl': table_corners[0], 'tr': table_corners[1],
                              'br': table_corners[2], 'bl': table_corners[3]}

        self.packing_side = packing_side
        self.frame_size = frame_size

        self.table_corner_distance_coeffs = {'tl': table_corners_distance_coeffs[0],
                                             'tr': table_corners_distance_coeffs[1],
                                             'br': table_corners_distance_coeffs[2],
                                             'bl': table_corners_distance_coeffs[3]}
        self.rect_table_corners = self.calculate_rect_table_area()

        self.rect_table_corner_distance_coeffs = self.calculate_rect_table_corner_distance_coeffs()

        self.rect_work_place_corners = self.define_work_place_corners()
        self.width = self.rect_table_corners['br'][0] - self.rect_table_corners['tl'][0]
        self.height = self.rect_table_corners['br'][1] - self.rect_table_corners['tl'][1]

        self.distance_coeff_width = abs(
            self.rect_table_corner_distance_coeffs['tr'] - self.rect_table_corner_distance_coeffs['tl'])
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

    def set_next_pack_task(self, next_pack_task: PackTask):
        self.next_pack_tasks = next_pack_task

    def define_work_place_corners(self):
        rect_table_corners_copy = copy.deepcopy(self.rect_table_corners)
        rect_table_corners_copy = {key: list(value) for key, value in rect_table_corners_copy.items()}
        rect_work_place_corners = calculate_work_place_size[self.packing_side](rect_table_corners_copy)

        rect_work_place_corners['tl'][0] = 0 if rect_work_place_corners['tl'][0] < 0 else rect_work_place_corners['tl'][
            0]
        rect_work_place_corners['tl'][1] = 0 if rect_work_place_corners['tl'][1] < 0 else rect_work_place_corners['tl'][
            1]
        rect_work_place_corners['br'][0] = self.frame_size[0] if rect_work_place_corners['br'][0] > self.frame_size[
            0] else \
            rect_work_place_corners['br'][0]
        rect_work_place_corners['br'][1] = self.frame_size[1] if rect_work_place_corners['br'][1] > self.frame_size[
            1] else \
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

    def detects_parts(self, object_shapes: list, precision_threshold=0.8):
        part_detections = []

        for pack_task in self.next_pack_tasks:
            acceptable_width_deviation = int(pack_task.part.width * (1 - precision_threshold))
            acceptable_width_range = (
                pack_task.part.width - acceptable_width_deviation, pack_task.part.width + acceptable_width_deviation)

            acceptable_height_deviation = int(pack_task.part.height * (1 - precision_threshold))
            acceptable_height_range = (
                pack_task.part.height - acceptable_height_deviation,
                pack_task.part.height + acceptable_height_deviation)

            founded_parts = []
            for object_shape in object_shapes:
                if (is_num_in_range(object_shape.width, acceptable_height_range) and
                        is_num_in_range(object_shape.height, acceptable_width_range)):
                    object_shape.width, object_shape.height = object_shape.height, object_shape.width

                elif not (is_num_in_range(object_shape.width, acceptable_width_range) and
                          is_num_in_range(object_shape.height, acceptable_height_range)):
                    break

                precision = calculate_two_sizes_match_precision((object_shape.width, object_shape.height),
                                                                (pack_task.part.width, pack_task.part.height))
                founded_parts.append((object_shape, precision))
            if founded_parts:
                for founded_part in founded_parts:
                    part_detections.append(PartDetection(pack_task.part, *founded_part))
        best_precision_detections = []
        used_parts = []
        used_object_shapes = []
        # sort by precision desc
        part_detections.sort(reverse=True, key=lambda detection: detection.precision)
        # not handled cases with equal precisions
        for part_detection in part_detections:
            the_part_amount = part_detection.part.multiplicity
            the_part_used_amount = sum([1 for used_part in used_parts if used_part is part_detection.part])
            if not (the_part_used_amount == the_part_amount or
                    part_detection.object_shape in used_object_shapes):
                best_precision_detections.append(part_detection)
                used_parts.append(part_detection.part)
                used_object_shapes.append(part_detection.object_shape)

        #self.update_part_detections(part_detections)

        return self.part_detections


    def update_part_detections(self, new_part_detections):
        old_parts = [part_detection.part for part_detection in self.part_detections]
        new_parts = [new_part_detection.part for new_part_detection in new_part_detections]
        for new_part_detection in new_part_detections:
            new_part_old_detections_indexes = [i for i, old_part in enumerate(old_parts) if new_part_detection.part is old_part]
            new_part_new_detection_indexes = [i for i, new_part in enumerate(new_parts) if new_part_detection.part is new_part]
            if len(new_part_old_detections_indexes) == 0:
                self.part_detections.append(new_part_detection)
            elif len(new_part_old_detections_indexes) == len(new_part_new_detection_indexes):
                for i in range(len(new_part_old_detections_indexes)):
                    self.part_detections[i] = new_part_detections[i]
            elif len(new_part_old_detections_indexes) > len(new_part_new_detection_indexes):
                correspondences = []
                for new_part_new_detection_index in new_part_new_detection_indexes:
                    for new_part_old_detections_index in new_part_old_detections_indexes:
                        old_part_detection = self.part_detections[new_part_old_detections_index]
                        new_part_detection = new_part_detections[new_part_new_detection_index]
                        correspondences.append((old_part_detection, new_part_detection, int(dist.euclidean(old_part_detection.object_shape.center,
                                                                                                           new_part_detection.object_shape.center))))

                correspondences.sort(key=lambda cr: cr[2])
                used_old_parts, used_new_parts = [], []
                for correspondence in correspondences:
                    old_part_detection = correspondence[0]
                    new_part_detection = correspondences[0]
                    #if not ()

def calculate_two_sizes_match_precision(matched_object: tuple, reference: tuple):
    width_diff = abs(matched_object[0] - reference[0])
    height_diff = abs(matched_object[1] - reference[1])

    if width_diff >= height_diff:
        precision = width_diff / reference[0]
    else:
        precision = height_diff / reference[1]
    return int((1 - precision) * 100) / 100
