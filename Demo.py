import numpy as np
import cv2
from collections import deque
import Utils
from WorkPlace import WorkPlace
from FRCNN import FRCNN
import Edging
from DataBase import DataBase
from PackTask import PackTask
import sys
from PIL import Image
from matplotlib import cm
from PIL import ImageFont, ImageDraw, Image
from datetime import datetime, timedelta
import argparse

def process_frame(frame):
    global previous_blurred_frame
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

    if previous_blurred_frame is None:
        previous_blurred_frame = blurred_frame
        return blurred_frame

    delta_frame = cv2.absdiff(previous_blurred_frame, blurred_frame)
    threshold_frame = cv2.threshold(delta_frame, 15, 255, cv2.THRESH_BINARY)[1]
    kernel = np.ones((5, 5), np.uint8)
    dilated_frame = cv2.dilate(threshold_frame, kernel, iterations=4)
    previous_blurred_frame = blurred_frame
    return dilated_frame


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
    return time.strftime(time_format)


def initialize_work_places():

    def set_work_place_task(work_place, task, next_task_time):
        work_place.set_cur_pack_task(task)
        work_place.set_next_pack_task_time(format_time_from_str(next_task_time))

    work_places = (WorkPlace('Муртазин Руслан Минислямович', ((563, 200), (1072, 200), (1057, 978), (505, 958)), 'Left',
                             (2.1, 2.2, 1.9, 1.8),
                             frame_size=(1920, 1080)),
                   WorkPlace('Бакшеев Александр Николаевич', ((1300, 214), (1605, 240), (1627, 1061), (1300, 1043)),
                             'Right', (2.2, 2.3, 2.0, 1.9),
                             frame_size=(1920, 1080)))

    cur_task, next_task_time = PackTask.get_pack_tasks(db, '01.07.2019 13:16',
                                                       work_places[0].packer)
    set_work_place_task(work_places[0], cur_task, next_task_time)

    cur_task, next_task_time = PackTask.get_pack_tasks(db, '01.07.2019 13:08',
                                                       work_places[1].packer)
    set_work_place_task(work_places[1], cur_task, next_task_time)

    return work_places


parser = argparse.ArgumentParser()
parser.add_argument('-i', dest='input_video', help='Input video file')
args = parser.parse_args()


CONTOUR_AREA_THRESHOLD = 3000
MINIMUM_DISTANCE_BETWEEN_RECTANGLES = 300
time_format = '%d.%m.%Y %H:%M:%S'

camera = cv2.VideoCapture(args.input_video)

cap_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
cap_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = camera.get(cv2.CAP_PROP_FPS) / 3
vid_writer = cv2.VideoWriter('edge_output.avi', cv2.VideoWriter_fourcc(*"XVID"), fps,
                             (cap_width, cap_height))
cv2.namedWindow("frame", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
previous_blurred_frame = None

db = DataBase()
current_time = format_time_from_str('01.07.2019 13:16:33')

work_places = initialize_work_places()
# bounding_areas = [list(work_place.rect_work_place_corners.values()) for work_place in work_places]

table_areas = [work_place.rect_table_corners for work_place in work_places]
# bounding_areas = Utils.combine_nearby_rects(bounding_areas)
# frcnn = FRCNN()

while True:
    captured, frame = camera.read()
    if not captured:
        break

    #cv2.imshow('part of table', frame[214:1061, 1300:1627])

    for work_place in work_places:
        if current_time == work_place.next_pack_task_time:
            cur_task, next_task_time = PackTask.get_pack_tasks(db, format_time_to_str(current_time)[:-3],
                                                               work_place.packer)
            work_place.set_cur_pack_task(cur_task)
            work_place.set_next_pack_task_time(format_time_from_str(next_task_time))
            work_place.reset_part_detections()

    # print out time
    cv2.putText(frame, format_time_to_str(current_time), (90, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

    for work_place in work_places:
        frame = work_place.apply_tasks_on_frame(frame)

    # processed_frame = process_frame(frame.copy())
    # contours, _ = cv2.findContours(processed_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # movement_rects = []
    #
    # for c in contours:
    #     if cv2.contourArea(c) < CONTOUR_AREA_THRESHOLD:
    #         continue
    #
    #     x, y, w, h = cv2.boundingRect(c)
    #     movement_rects.append(((x, y), (x + w, y + h)))

    # movement_rects = Utils.combine_nearby_rects(movement_rects, shift=MINIMUM_DISTANCE_BETWEEN_RECTANGLES)
    # movement_rects = Utils.restrict_rectangle_areas_by_another_ones(movement_rects, table_areas)

    # for movement_rect in movement_rects:
    #     roi = frame[movement_rect[0][1]:movement_rect[1][1], movement_rect[0][0]:movement_rect[1][0]]
    # frcnn.forward(roi)

    table_views = [frame[table_area['tl'][1]:table_area['br'][1], table_area['tl'][0]:table_area['br'][0]] for
                   table_area in table_areas]
    all_objects_shapes = []
    for table_view in table_views:
        all_objects_shapes.append(Edging.get_clockwise_midside_points(table_view))

    for i, table_object_shapes in enumerate(all_objects_shapes):
        table_view = table_views[i]
        work_place = work_places[i]
        part_detections = work_place.detects_parts(table_object_shapes)
        work_place.visualize_part_detections(frame)

        # for object_shape in table_object_shapes:
        #     Edging.apply_info(object_shape, table_view, work_place)
    cv2.imshow('frame', frame)
    vid_writer.write(frame)
    cv2.waitKey(1)
    update_time()

camera.release()
vid_writer.release()
cv2.destroyAllWindows()
