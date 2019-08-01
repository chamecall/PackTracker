import numpy as np
import cv2
from collections import deque
import Utils
from WorkPlace import WorkPlace
from FRCNN import FRCNN
from Edging import find_contours
from DataBase import DataBase
from PackTask import PackTask
import sys
from PIL import Image
from matplotlib import cm
from PIL import ImageFont, ImageDraw, Image
from datetime import datetime, timedelta
import argparse


def update_time():
    global fps, current_time
    update_time.frame_num += 1

    if update_time.frame_num >= fps:
        current_time += timedelta(seconds=1)
        update_time.frame_num = update_time.frame_num % fps


update_time.frame_num = 0


def format_time_from_str(str_time):
    return datetime.strptime(str_time, time_format)


def format_time_to_str(time: datetime):
    return f'{time.month}/{time.day}/{time.year} {time.hour:02}:{time.minute:02}:{time.second:02}'


def initialize_work_places():
    def set_work_place_task(work_place, task, next_task_time):
        work_place.set_cur_pack_task(task)
        work_place.set_next_pack_task_time(format_time_from_str(next_task_time))

    work_places = (WorkPlace('Муртазин Руслан Минислямович', ((600, 200), (1072, 200), (1057, 978), (542, 958)), 'Left',
                             (2.1, 2.2, 1.9, 1.8),
                             frame_size=(1920, 1080)),
                   WorkPlace('Бакшеев Александр Николаевич', ((1300, 214), (1605, 240), (1627, 1061), (1300, 1043)),
                             'Right', (2.2, 2.3, 2.0, 1.9),
                             frame_size=(1920, 1080)))

    cur_task, next_task_time = PackTask.get_pack_tasks(db, '7/1/2019 13:16:42',
                                                       work_places[0].packer)
    set_work_place_task(work_places[0], cur_task, next_task_time)

    cur_task, next_task_time = PackTask.get_pack_tasks(db, '7/1/2019 13:08:23',
                                                       work_places[1].packer)
    set_work_place_task(work_places[1], cur_task, next_task_time)

    return work_places


def get_bounding_boxes_from_contours(contours):
    bounding_boxes = []
    for c in contours:
        if cv2.contourArea(c) < CONTOUR_AREA_THRESHOLD:
            continue
        x, y, w, h = cv2.boundingRect(c)
        bounding_boxes.append(((x, y), (x + w, y + h)))
        return bounding_boxes


parser = argparse.ArgumentParser()
parser.add_argument('-i', dest='input_video', help='Input video file')
args = parser.parse_args()

CONTOUR_AREA_THRESHOLD = 3000
MINIMUM_DISTANCE_BETWEEN_RECTANGLES = 300
time_format = '%m/%d/%Y %H:%M:%S'

camera = cv2.VideoCapture(args.input_video)

cap_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
cap_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = camera.get(cv2.CAP_PROP_FPS)
vid_writer = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*"FMP4"), fps,
                             (cap_width, cap_height))
cv2.namedWindow("frame", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

db = DataBase()
current_time = format_time_from_str('7/1/2019 13:16:33')
work_places = initialize_work_places()

# frcnn = FRCNN()

while True:
    captured, frame = camera.read()
    if not captured:
        break

    for work_place in work_places:
        table_part_of_frame = work_place.get_table_view_from_frame(frame)

        if current_time == work_place.next_pack_task_time:
            cur_task, next_task_time = PackTask.get_pack_tasks(db, format_time_to_str(current_time),
                                                               work_place.packer)
            work_place.set_cur_pack_task(cur_task)
            work_place.set_next_pack_task_time(format_time_from_str(next_task_time))
            work_place.reset_pack_task()

        # cv2.imshow(f'{work_place.packer}', table_part_of_frame)
        # cv2.waitKey(1)

        movement_frame = work_place.get_movement_area(frame)
        contours, _ = cv2.findContours(movement_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        movement_rects = get_bounding_boxes_from_contours(contours)
        movement_rects = Utils.combine_nearby_rects(movement_rects, shift=MINIMUM_DISTANCE_BETWEEN_RECTANGLES)
        table_object_shapes = []

        for movement_rect in movement_rects:
            rom = table_part_of_frame[movement_rect[0][1]:movement_rect[1][1], movement_rect[0][0]:movement_rect[1][0]]
            table_object_shapes += find_contours(rom, movement_rect[0])
        work_place.detects_parts(table_part_of_frame, table_object_shapes)
        work_place.visualize_part_detections(frame)
        frame = work_place.apply_tasks_on_frame(frame)

    # print out our time
    cv2.putText(frame, format_time_to_str(current_time), (90, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

    # for movement_rect in movement_rects:
    #     roi = frame[movement_rect[0][1]:movement_rect[1][1], movement_rect[0][0]:movement_rect[1][0]]
    # frcnn.forward(roi)

    cv2.imshow('frame', frame)
    vid_writer.write(frame)
    cv2.waitKey(1)
    update_time()

camera.release()
vid_writer.release()
cv2.destroyAllWindows()
