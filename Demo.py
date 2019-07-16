import numpy as np
import cv2
from collections import deque
import Utils
from WorkPlace import WorkPlace
from Yolo import Yolo

CONTOUR_AREA_THRESHOLD = 3000
MINIMUM_DISTANCE_BETWEEN_RECTANGLES = 300
camera = cv2.VideoCapture('/home/algernon/samba/video_queue/omega-packaging/data/raw/boxes.avi')
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
previous_blurred_frame = None

work_places = (WorkPlace('Incognito', ((500, 100), (1100, 200)), 'Left', frame_size=(1920, 1080)),
               WorkPlace('Noname', ((1100, 100), (1650, 1080)), 'Right', frame_size=(1920, 1080)))

bounding_areas = [work_place.work_place_position for work_place in work_places]
bounding_areas = Utils.combine_nearby_rects(bounding_areas)
yolo = Yolo()


while True:
    captured, frame = camera.read()
    if not captured:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)
    if previous_blurred_frame is None:
        previous_blurred_frame = blurred_frame
        continue

    delta_frame = cv2.absdiff(previous_blurred_frame, blurred_frame)
    threshold_frame = cv2.threshold(delta_frame, 15, 255, cv2.THRESH_BINARY)[1]
    kernel = np.ones((5, 5), np.uint8)
    dilated_frame = cv2.dilate(threshold_frame, kernel, iterations=4)
    contours, _ = cv2.findContours(dilated_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rectangles = []

    for c in contours:
        if cv2.contourArea(c) < CONTOUR_AREA_THRESHOLD:
            continue

        x, y, w, h = cv2.boundingRect(c)
        rectangles.append(((x, y), (x + w, y + h)))

    rectangles = Utils.combine_nearby_rects(rectangles, shift=MINIMUM_DISTANCE_BETWEEN_RECTANGLES)
    rectangles = Utils.restrict_rectangle_areas_by_another_ones(rectangles, bounding_areas)

    for rectangle in rectangles:
        #cv2.rectangle(frame, rectangle[0], rectangle[1], (0, 255, 0), 2)
        detections = yolo.forward(frame[rectangle[0][1]:rectangle[1][1], rectangle[0][0]:rectangle[1][0]])
        #print(detections)
    previous_blurred_frame = blurred_frame

    #cv2.imshow("Frame3: Delta", delta_frame)
    #cv2.imshow("Frame6: Contours", frame)
    # cv2.imshow('Table1 only', frame[100:1080, 200:1100])
    # cv2.imshow('Table2 only', frame[100:1080, 1100:1950])

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key != 255:
        print('key:', [chr(key)])

camera.release()
cv2.destroyAllWindows()
