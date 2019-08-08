import numpy as np
import cv2
from collections import deque
import Utils
from WorkPlace import WorkPlace
#from darknet_video import YOLO
from Edging import find_contours
from DataBase import DataBase
from PackTask import PackTask
import sys
from PIL import Image
from matplotlib import cm
from PIL import ImageFont, ImageDraw, Image
from datetime import datetime, timedelta
import argparse
from Utils import set_better_channel
from Detector import Detector


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

    work_places = (WorkPlace('Муртазин Руслан Минислямович', ((560, 200), (1072, 200), (1057, 978), (502, 958)), 'Left',
                             (2.1, 2.2, 1.9, 1.8),
                             frame_size=(1920, 1080)),
                   WorkPlace('Бакшеев Александр Николаевич', ((1150, 214), (1605, 240), (1627, 1061), (1150, 1043)),
                             'Right', (2.2, 2.3, 2.0, 1.9),
                             frame_size=(1920, 1080))
                   )

    cur_task, next_task_time = PackTask.get_pack_tasks(db, '7/1/2019 13:16:42',
                                                       work_places[0].packer)
    print(cur_task)
    set_work_place_task(work_places[0], cur_task, next_task_time)

    cur_task, next_task_time = PackTask.get_pack_tasks(db, '7/1/2019 13:17:04',
                                                       work_places[1].packer)
    print(cur_task)
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
length = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))
print( length )

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

box_detector = Detector('/home/algernon/PycharmProjects/test/boxes_detections.json')

part_detector = Detector('/home/algernon/PycharmProjects/test/parts_detections.json')
print(len(part_detector.detections))
while True:
    captured, frame = camera.read()
    if not captured:
        break
    box_detections = box_detector.get_detections_per_frame()
    part_detections = part_detector.get_detections_per_frame()
    for work_place in work_places:

        if work_place.next_pack_task_time and \
                current_time >= work_place.next_pack_task_time:
            cur_task, next_task_time = PackTask.get_pack_tasks(db, format_time_to_str(current_time),
                                                                        work_place.packer)
            work_place.set_cur_pack_task(cur_task)
            work_place.set_next_pack_task_time(format_time_from_str(next_task_time))
            work_place.reset_pack_task()

        # cv2.imshow(f'{work_place.packer}', table_part_of_frame)
        # cv2.waitKey(1)

        work_place.detect_parts(frame, part_detections)
        work_place.detect_boxes(frame, box_detections)

        # work_place.visualize_part_detections(frame)
        work_place.visualize_box_detections(frame)
        frame = work_place.apply_tasks_on_frame(frame)

    # print out our time
    cv2.putText(frame, format_time_to_str(current_time), (90, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

    cv2.imshow('frame', frame)
    vid_writer.write(frame)
    cv2.waitKey(1)
    update_time()

camera.release()
vid_writer.release()
cv2.destroyAllWindows()
