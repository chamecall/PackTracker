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





CONTOUR_AREA_THRESHOLD = 3000
MINIMUM_DISTANCE_BETWEEN_RECTANGLES = 300
camera = cv2.VideoCapture(
    '/home/chame/Desktop/МЕЛ 1 стол упаковки 1 _20190701-124357--20190701-144357_EDIT.avi')

cap_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
cap_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = camera.get(cv2.CAP_PROP_FPS)
vid_writer = cv2.VideoWriter('edge_output.avi', cv2.VideoWriter_fourcc(*"XVID"), fps / 3,
                             (cap_width, cap_height))
cv2.namedWindow("frame", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
previous_blurred_frame = None

work_places = (WorkPlace('Муртазин Руслан Минислямович', ((563, 200), (1072, 200), (1057, 978), (505, 958)), 'Left',
                         (2.1, 2.2, 1.9, 1.8),
                         frame_size=(1920, 1080)),
               WorkPlace('Бакшеев Александр Николаевич', ((1132, 214), (1605, 240), (1627, 1061), (1146, 1043)),
                         'Right', (2.2, 2.3, 2.0, 1.9),
                         frame_size=(1920, 1080)))

db = DataBase()
initial_time = '01.07.2019 13:16'

for work_place in work_places:
    work_place.set_next_pack_task(PackTask.get_pack_tasks(db, initial_time, work_place.packer))


bounding_areas = [list(work_place.rect_work_place_corners.values()) for work_place in work_places]

table_areas = [work_place.rect_table_corners for work_place in work_places]
bounding_areas = Utils.combine_nearby_rects(bounding_areas)
# frcnn = FRCNN()

while True:
    captured, frame = camera.read()
    if not captured:
        break

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

camera.release()
vid_writer.release()
cv2.destroyAllWindows()
