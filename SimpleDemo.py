import numpy as np
import cv2
import Utils
from WorkPlace import WorkPlace
from Yolo import Yolo, cv_draw_boxes



camera = cv2.VideoCapture('/home/algernon/samba/video_queue/omega-packaging/data/raw/boxes.avi')
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

yolo = Yolo()


while True:
    captured, frame = camera.read()
    if not captured:
        break

    detections = yolo.forward(frame)
    image = cv_draw_boxes(detections, frame)
    cv2.imshow('Result', image)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key != 255:
        print('key:', [chr(key)])

camera.release()
cv2.destroyAllWindows()
