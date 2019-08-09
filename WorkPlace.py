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
from Utils import are_rectangles_intersect, rectangles_intersection, rect_square, is_point_in_rect
from Hands import Hands

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
    WORKER_PLACE_SIZE = 200

    def calculate_rect_table_area(self):
        tcs = self.table_corners
        min_x, min_y = min(tcs['tl'][0], tcs['bl'][0]), min(tcs['tl'][1], tcs['tr'][1])
        max_x, max_y = max(tcs['tr'][0], tcs['br'][0]), max(tcs['bl'][1], tcs['br'][1])
        rect_table_corners = {'tl': (min_x, min_y),
                              'br': (max_x, max_y)}
        return rect_table_corners

    font = ImageFont.truetype("arial.ttf", 16)
    bold_font = ImageFont.truetype("arial_bold.ttf", 16)
    large_font = ImageFont.truetype("arial.ttf", 20)
    line_height_size = 20

    def get_table_view_from_frame(self, frame: np.ndarray):
        table_rect = self.rect_table_corners
        return frame[table_rect['tl'][1]:table_rect['br'][1], table_rect['tl'][0]:table_rect['br'][0]]

    def get_work_place_view_from_frame(self, frame: np.ndarray):
        work_pl_rect = self.rect_work_place_corners
        return frame[work_pl_rect['tl'][1]:work_pl_rect['br'][1], work_pl_rect['tl'][0]:work_pl_rect['br'][0]]

    def __init__(self, packer, table_corners: tuple, packing_side,
                 frame_size=(1366, 768)):
        self.worker_hands = Hands()
        self.all_parts_are_in_box = False
        self.pack_task_completed = False
        self.part_detections = []
        self.box_detection = None
        self.previous_blurred_table = None
        self.part_tracker = ObjectTracker()
        self.part_detector = PartDetector()
        self.box_detector = BoxDetector()
        self.opened_box_tracker = ObjectTracker(distance_threshold=150)
        self.closed_box_tracker = ObjectTracker()
        self.next_pack_task_time = None
        self.cur_pack_task = None
        self.table_corners = {'tl': table_corners[0], 'tr': table_corners[1],
                              'br': table_corners[2], 'bl': table_corners[3]}
        self.packer = packer
        self.packing_side = packing_side
        self.frame_size = frame_size

        self.rect_table_corners = self.calculate_rect_table_area()

        self.rect_work_place_corners = self.define_work_place_corners()
        self.width = self.rect_table_corners['br'][0] - self.rect_table_corners['tl'][0]
        self.height = self.rect_table_corners['br'][1] - self.rect_table_corners['tl'][1]

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
        if not parts_detections or self.all_parts_are_in_box:
            return

        frame = self.get_work_place_view_from_frame(frame)
        parts_detections = [detection for detection in parts_detections if self.is_rect_in_work_place(detection[2])]
        best_detections = self.part_detector.detect(self.cur_pack_task, parts_detections)
        table_best_detections = []
        for best_detection in best_detections:
            best_detection.object_shape.set_box_rect(self.transform_coords_in_work_place_axis(best_detection.object_shape.box_rect))
            table_best_detections.append(best_detection)

        self.part_detections = self.part_tracker.track(frame, table_best_detections)
        self.update_undetected_pack_items()
        self.update_detected_pack_items()

    def update_detected_pack_items(self):

        in_box_part_detections = [(i, part_detection) for i, part_detection in enumerate(self.part_detections) if
                                  part_detection.is_in_box()]

        if len(in_box_part_detections) == len(self.cur_pack_task):
            self.all_parts_are_in_box = True
            return

        tracked_part_detections = [(i, part_detection) for i, part_detection in enumerate(self.part_detections) if
                                   part_detection.is_tracked()]

        moved_part_detections = [(i, part_detection) for i, part_detection in enumerate(self.part_detections) if
                                 part_detection.is_moved()]

        if self.box_detection:
            box_rect = self.box_detection.object_shape.tl_box_rect
            for i, tracked_part_detection in tracked_part_detections:
                part_box_rect = tracked_part_detection.object_shape.tl_box_rect
                if are_rectangles_intersect(box_rect, part_box_rect):
                    percent_intersection = int(rect_square(rectangles_intersection(box_rect, part_box_rect)) / rect_square(part_box_rect) * 100)
                    # 30 is an intersection of the part and the box in percent to consider part inside of the box
                    print(percent_intersection)

                    if percent_intersection > 30:
                        self.part_detections[i].set_status_as_in_box()

    def update_undetected_pack_items(self):
        pack_task_parts = [pack_task_item.part for pack_task_item in self.cur_pack_task]
        part_detections_parts = set(part_detection.part for part_detection in self.part_detections)
        for i, pack_task_part in enumerate(pack_task_parts):
            if pack_task_part in part_detections_parts:
                self.cur_pack_task[i].set_status_as_detected()
            else:
                self.cur_pack_task[i].set_status_as_not_detected()

    def visualize_part_detections(self, frame):

        for i, part_detection in enumerate(self.part_tracker.get_detections()):
            shape = part_detection.object_shape
            points = [shape.box_rect_center, shape.tm_point_rect, shape.rm_point_rect]
            points = [(point[0] + self.rect_work_place_corners['tl'][0],
                       point[1] + self.rect_work_place_corners['tl'][1]) for point in points]

            print_pos, index = \
                [(next_pack_task.print_pos, next_pack_task.index) for next_pack_task in self.cur_pack_task if
                 next_pack_task.part is part_detection.part][0]

            color = self.generate_color_by_index(index)

            box_rect = tuple((shape.box_rect[0] + self.rect_work_place_corners['tl'][0],
                              shape.box_rect[1] + self.rect_work_place_corners['tl'][1], *shape.box_rect[2:]))

            cv2.rectangle(frame, box_rect, color, 2)

            def putText(x, y, value: float):
                cv2.putText(frame, f'{int(value)} px', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

            putText(*points[0], sum(self.part_tracker.shift_histories[i]))

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

        def draw_name_info(x, y, name, pack_number, color):
            font = WorkPlace.large_font
            draw.text((x, y), f"{name} - {pack_number}", color, font=font)

        x_value, y_value = None, None
        name = ''
        if self.packing_side == 'Left':
            name = 'Бакшеев Александр Николаевич'
            y_value = 110
            x_value = 5
        elif self.packing_side == 'Right':
            name = 'Муртазин Руслан Минислямович'
            y_value = 110
            x_value = frame.shape[1] - 450

        pack_number = self.cur_pack_task[0].part.pack_number
        draw_name_info(x_value, 10, name, pack_number, (0, 255, 255))

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



    def visualize_box_detections(self, frame):
        table_part_of_frame = self.get_work_place_view_from_frame(frame)

        def draw_box(box, color):
            cv2.rectangle(table_part_of_frame, box.object_shape.box_rect, color, 2)
            cv2.putText(table_part_of_frame, f'{PackBox.reversed_statuses[box.status]} ({box.probability}%)',
                        box.object_shape.box_rect_center, cv2.FONT_HERSHEY_COMPLEX, 0.7,
                        color, 2)

        for box in self.opened_box_tracker.detections:
            draw_box(box, (255, 255, 255))

        for box in self.closed_box_tracker.detections:
            draw_box(box, (0, 0, 255))

    def detect_opened_boxes(self, frame, opened_box_detections):
        pack_box = self.get_pack_box_from_boxes(opened_box_detections, PackBox.statuses['Opened'], 0.15)
        return self.opened_box_tracker.track(frame, pack_box)

    def detect_closed_boxes(self, frame, closed_box_detections):
        pack_box = self.get_pack_box_from_boxes(closed_box_detections, PackBox.statuses['Closed'], 0)
        return self.closed_box_tracker.track(frame, pack_box)

    def get_pack_box_from_boxes(self, box_detections, status, probability):
        opened_box_detections = [detection for detection in box_detections if
                                 self.is_rect_in_work_place(detection[2])]
        pack_box = []
        if opened_box_detections:
            best_box_detection = max(opened_box_detections, key=lambda detection: detection[1])
            if best_box_detection[1] > probability:
                best_box_rect = self.transform_coords_in_work_place_axis(best_box_detection[2])
                pack_box = [PackBox(ObjectShape(best_box_rect), status, int(best_box_detection[1] * 100))]
        return pack_box

    def is_rect_in_work_place(self, rect):
        tb = self.rect_work_place_corners

        xs = rect[0] - rect[2] / 2, rect[0] + rect[2] / 2
        ys = rect[1] - rect[3] / 2, rect[1] + rect[3] / 2
        return all([tb['tl'][0] <= x <= tb['br'][0] for x in xs]) and all([tb['tl'][1] <= y <= tb['br'][1] for y in ys])

    def is_point_in_work_place(self, point):
        tb = self.rect_work_place_corners
        return is_num_in_range(point[0], (tb['tl'][0], tb['br'][0])) and is_num_in_range(point[1],
                                                                                       (tb['tl'][1], tb['br'][1]))

    def transform_coords_in_work_place_axis(self, rect):
        tb = self.rect_work_place_corners
        new_rect = (rect[0] - tb['tl'][0], rect[1] - tb['tl'][1], *rect[2:])
        return new_rect

    def detect_boxes(self, frame, boxes_detections):
        frame = self.get_work_place_view_from_frame(frame)

        if not self.box_detection:
            boxes_detections = tuple(detection for detection in boxes_detections if detection[0] == 'pb_open')
            result = self.detect_opened_boxes(frame, boxes_detections)
            self.box_detection = result[0] if result else None

        elif self.box_detection.status == PackBox.statuses['Opened'] and not self.all_parts_are_in_box:
            result = self.opened_box_tracker.update(frame)
            self.box_detection = result[0] if result else None

        elif self.box_detection.status == PackBox.statuses['Opened'] and self.all_parts_are_in_box:
            boxes_detections = tuple(detection for detection in boxes_detections if detection[0] == 'pb_closed')
            box_is_closed = self.detect_closed_boxes(frame, boxes_detections)
            if box_is_closed:
                self.pack_task_completed = True

    def visualize_hand_detections(self, frame):
        self.worker_hands.draw_skeleton(frame)

    def detect_hands(self, hands_detections):
        for hands_detection in hands_detections:
            for point in hands_detection.values():
                if not point:
                    continue
                if self.is_point_in_work_place(point):
                    self.worker_hands.set_new_points(hands_detection)
                break
