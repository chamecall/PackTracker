import copy
import math
import cv2
import PackTask
from PartDetection import PartDetection
from Utils import is_num_in_range, calculate_two_sizes_match_precision
from scipy.spatial import distance as dist
from PIL import ImageFont, ImageDraw, Image
import numpy as np
from ObjectTracker import ObjectTracker
from PartDetector import PartDetector
from BoxDetector import BoxDetector
from PackBox import PackBox
from ObjectShape import ObjectShape


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
    WORKER_PLACE_SIZE = 100

    def calculate_rect_table_area(self):
        tcs = self.table_corners
        min_x, min_y = min(tcs['tl'][0], tcs['bl'][0]), min(tcs['tl'][1], tcs['tr'][1])
        max_x, max_y = max(tcs['tr'][0], tcs['br'][0]), max(tcs['bl'][1], tcs['br'][1])
        rect_table_corners = {'tl': (min_x, min_y),
                              'br': (max_x, max_y)}
        return rect_table_corners

    font = ImageFont.truetype("arial.ttf", 16)
    bold_font = ImageFont.truetype("arial_bold.ttf", 16)
    line_height_size = 20

    def get_table_view_from_frame(self, frame: np.ndarray):
        table_rect = self.rect_table_corners
        return frame[table_rect['tl'][1]:table_rect['br'][1], table_rect['tl'][0]:table_rect['br'][0]]

    def get_work_place_view_from_frame(self, frame: np.ndarray):
        work_pl_rect = self.rect_work_place_corners
        return frame[work_pl_rect['tl'][1]:work_pl_rect['br'][1], work_pl_rect['tl'][0]:work_pl_rect['br'][0]]

    def __init__(self, packer, table_corners: tuple, packing_side, table_corners_distance_coeffs,
                 frame_size=(1366, 768)):
        self.previous_blurred_table = None
        self.part_tracker = ObjectTracker()
        self.part_detector = PartDetector()
        self.box_detector = BoxDetector()
        self.opened_box_tracker = ObjectTracker(distance_threshold=150)
        self.closed_box_tracker = ObjectTracker()
        self.next_pack_task_time = None
        self.cur_pack_task = None
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

    def set_cur_pack_task(self, cur_pack_task):
        self.cur_pack_task = cur_pack_task

    def set_next_pack_task_time(self, next_pack_task_time):
        self.next_pack_task_time = next_pack_task_time

    def reset_pack_task(self):
        self.part_tracker = ObjectTracker()

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

    def detect_parts(self, frame, parts_detections: list):
        if not parts_detections:
            return
        best_precision_part_detections = self.part_detector.detect(self.cur_pack_task, parts_detections)
        frame = self.get_work_place_view_from_frame(frame)
        self.move_detection_coords_to_table_area(best_precision_part_detections)
        self.part_tracker.track(frame, best_precision_part_detections)
        self.update_pack_task_statuses()

    def move_detection_coords_to_table_area(self, detections):
        for detection in detections:
            detection.object_shape.box_rect = self.transform_coords_in_table_axis(detection.object_shape.box_rect)


    def update_pack_task_statuses(self):
        pack_task_parts = [pack_task_item.part for pack_task_item in self.cur_pack_task]
        part_detections_parts = set(part_detection.part for part_detection in self.part_tracker.get_detections())
        for i, pack_task_part in enumerate(pack_task_parts):
            if pack_task_part in part_detections_parts:
                self.cur_pack_task[i].set_status_as_detected()
            else:
                self.cur_pack_task[i].set_status_as_not_detected()

    def visualize_part_detections(self, frame):

        for part_detection in self.part_tracker.get_detections():
            shape = part_detection.object_shape
            points = [shape.box_rect_center, shape.tm_point_rect, shape.rm_point_rect]


            print_pos, index = \
                [(next_pack_task.print_pos, next_pack_task.index) for next_pack_task in self.cur_pack_task if
                 next_pack_task.part is part_detection.part][0]

            color = self.generate_color_by_index(index)

            cv2.rectangle(frame, shape.box_rect, (0, 0, 255), 2)

            def putText(x, y, value: float):
                cv2.putText(frame, f'{int(value)} mm', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)

            putText(points[1][0] - 15, points[1][1] - 10, shape.box_rect[3])
            putText(points[2][0] + 10, points[2][1], shape.box_rect[2])
            print(print_pos)
            print(index)
            print_pos = (print_pos[0] - 3, int(print_pos[1] + WorkPlace.line_height_size / 2 - 1))
            cv2.circle(frame, print_pos, 3, color, -1)
            cv2.circle(frame, points[0], 3, color, -1)
            cv2.line(frame, print_pos, points[0],
                     color, 2)

    def apply_tasks_on_frame(self, frame):
        im = Image.fromarray(frame)
        draw = ImageDraw.Draw(im)

        def draw_text(x, y, task, color, is_bold=False):
            font = WorkPlace.bold_font if is_bold else WorkPlace.font
            draw.text((x, y), f"{task.part.name} ({task.amount}), "
            f"{task.part.height}x{task.part.width}x{task.part.depth}", color, font=font)

        x_value, y_value = None, None
        if self.packing_side == 'Left':
            y_value = 0
            x_value = 5
        elif self.packing_side == 'Right':
            y_value = 0
            x_value = frame.shape[1] - 350

        for cur_task in self.cur_pack_task:
            bold = False
            color = self.generate_color_by_index(cur_task.index) if cur_task.is_detected() else (0, 0, 0)
            if cur_task.is_detected():
                bold = True
            draw_text(x_value, y_value, cur_task, color, is_bold=bold)
            cur_task.print_pos = (x_value, y_value)
            y_value += WorkPlace.line_height_size

        frame = np.asarray(im)
        return frame

    def generate_color_by_index(self, index):
        tasks_num = len(self.cur_pack_task)
        color_step = int(255 / (tasks_num / 3))
        bgr = 255, 255, 255
        lvl = index // 3
        bgr = [channel - lvl * color_step for channel in bgr]
        bgr[index % 3] -= color_step
        return tuple(bgr)

    def get_movement_area(self, frame):
        frame = self.get_table_view_from_frame(frame)

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        blurred_frame = cv2.GaussianBlur(gray_frame, (7, 7), 0)

        if self.previous_blurred_table is None:
            self.previous_blurred_table = blurred_frame
            return np.zeros(frame.shape[:-1]).astype('uint8')

        delta_frame = cv2.absdiff(self.previous_blurred_table, blurred_frame)

        threshold_frame = cv2.threshold(delta_frame, 15, 255, cv2.THRESH_BINARY)[1]

        kernel = np.ones((3, 3), np.uint8)
        dilated_frame = cv2.dilate(threshold_frame, kernel, iterations=3)
        # cv2.imshow(f'{self.packer}', dilated_frame)
        # cv2.waitKey(1)
        self.previous_blurred_table = blurred_frame

        return dilated_frame

    def visualize_box_detections(self, frame):
        table_part_of_frame = self.get_work_place_view_from_frame(frame)
        for box in self.opened_box_tracker.detections:
            cv2.rectangle(table_part_of_frame, box.object_shape.box_rect, (255, 0, 0), 2)
            cv2.putText(table_part_of_frame, f'{PackBox.reversed_statuses[box.status]}',
                        box.object_shape.box_rect_center, cv2.FONT_HERSHEY_COMPLEX, 0.7,
                        (255, 0, 0), 2)

    def detect_opened_boxes(self, frame, opened_box_detections):
        work_place_area = self.get_work_place_view_from_frame(frame)
        pack_box = self.get_pack_box_from_boxes(opened_box_detections, PackBox.statuses['Opened'])
        self.opened_box_tracker.track(work_place_area, pack_box)

    def detect_closed_boxes(self, frame, closed_box_detections):
        work_place_area = self.get_work_place_view_from_frame(frame)
        pack_box = self.get_pack_box_from_boxes(closed_box_detections, PackBox.statuses['Closed'])
        self.opened_box_tracker.track(work_place_area, pack_box)

    def get_pack_box_from_boxes(self, box_detections, status):
        opened_box_detections = [detection for detection in box_detections if
                                 self.is_rect_on_the_table(detection[2])]
        pack_box = []
        if opened_box_detections:
            best_box_detection = max(opened_box_detections, key=lambda detection: detection[1])[2]
            best_box_detection = self.transform_coords_in_table_axis(best_box_detection)

            pack_box = [PackBox(ObjectShape(best_box_detection), status)]
        return pack_box

    def is_rect_on_the_table(self, rect):
        tb = self.rect_work_place_corners
        print(tb)
        print(rect)
        xs = rect[0] - rect[2] / 2, rect[0] + rect[2] / 2
        ys = rect[1] - rect[3] / 2, rect[1] + rect[3] / 2
        return all([tb['tl'][0] <= x <= tb['br'][0] for x in xs]) and all([tb['tl'][1] <= y <= tb['br'][1] for y in ys])

    def transform_coords_in_table_axis(self, rect):
        tb = self.rect_work_place_corners
        new_rect = (rect[0] - tb['tl'][0], rect[1] - tb['tl'][1], *rect[2:])
        return new_rect

    def detect_boxes(self, frame, boxes_detections):
        if not self.opened_box_tracker.detections:
            boxes_detections = tuple(detection for detection in boxes_detections if detection[0] == 'pb_open')
            self.detect_opened_boxes(frame, boxes_detections)
        else:
            boxes_detections = tuple(detection for detection in boxes_detections if detection[0] == 'pb_closed')
            self.detect_closed_boxes(frame, boxes_detections)
