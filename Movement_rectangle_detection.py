import numpy as np
import cv2
from collections import deque

CONTOUR_AREA_THRESHOLD = 3000
MINIMUM_DISTANCE_BETWEEN_RECTANGLES = 300
camera = cv2.VideoCapture('/home/algernon/samba/video_queue/omega-packaging/data/raw/boxes.avi')
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
previous_blurred_frame = None


def are_rectangles_beside(first_rectangle, second_rectangle):
    if first_rectangle[0][0] > second_rectangle[1][0] + MINIMUM_DISTANCE_BETWEEN_RECTANGLES or second_rectangle[0][0] > \
            first_rectangle[1][0] + MINIMUM_DISTANCE_BETWEEN_RECTANGLES:
        return False

    if first_rectangle[0][1] > second_rectangle[1][1] + MINIMUM_DISTANCE_BETWEEN_RECTANGLES or second_rectangle[0][1] > \
            first_rectangle[1][1] + MINIMUM_DISTANCE_BETWEEN_RECTANGLES:
        return False
    return True


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
    rectangles = deque()

    for c in contours:
        if cv2.contourArea(c) < CONTOUR_AREA_THRESHOLD:
            continue

        x, y, w, h = cv2.boundingRect(c)
        rectangles.append(((x, y), (x + w, y + h)))

    num_permutations = 0
    epochs = 0
    base_rectangle = rectangles.popleft() if rectangles else None
    while rectangles:
        rectangle = rectangles.popleft()
        if are_rectangles_beside(base_rectangle, rectangle):
            base_rectangle = ((min(base_rectangle[0][0], rectangle[0][0]), min(base_rectangle[0][1], rectangle[0][1])),
                              (max(base_rectangle[1][0], rectangle[1][0]), max(base_rectangle[1][1], rectangle[1][1])))
            num_permutations = 0
        else:
            rectangles.append(rectangle)
            num_permutations += 1

        num_rectangles = len(rectangles)
        if num_permutations >= num_rectangles:
            rectangles.append(base_rectangle)
            base_rectangle = rectangles.popleft()
            num_permutations = 0
            epochs += 1

        if epochs == num_rectangles:
            rectangles.append(base_rectangle)
            break

    while rectangles:
        rectangle = rectangles.pop()
        cv2.rectangle(frame, rectangle[0], rectangle[1], (0, 255, 0), 2)
    previous_blurred_frame = blurred_frame

    cv2.imshow("Frame3: Delta", delta_frame)
    cv2.imshow("Frame6: Contours", frame)
    cv2.imshow('Table only', frame[100:1080, 200:1950])


    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key != 255:
        print('key:', [chr(key)])

camera.release()
cv2.destroyAllWindows()
