from __future__ import print_function
import cv2 as cv2
import argparse
from Edging import find_contours
from ObjectShape import ObjectShape

import imutils
import numpy as np
from imutils import perspective
from imutils import contours


parser = argparse.ArgumentParser(description='Code for Thresholding Operations using inRange tutorial.')
parser.add_argument('--video', help='asdf.')
args = parser.parse_args()

## [cap]
cap = cv2.VideoCapture(args.video)
## [cap]
cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
vid_writer = cv2.VideoWriter('boxes.avi', cv2.VideoWriter_fourcc(*"FMP4"), fps,
                         (cap_width, cap_height))



window_detection_name = 'boxes'
cv2.namedWindow(window_detection_name, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(window_detection_name,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)



while True:

    ret, orig = cap.read()
    if orig is None:
        break

    frame = cv2.cvtColor(orig, cv2.COLOR_BGR2HSV)
    frame = cv2.GaussianBlur(frame, (7, 7), 0)
    frame_box_out = cv2.inRange(frame, (0, 20, 150), (20, 115, 230))
    frame_box_in = cv2.inRange(frame, (12, 86, 67), (23, 154, 123))
    frame_box_out = cv2.dilate(frame_box_out, (3, 3), iterations=1)
    frame_box_in = cv2.dilate(frame_box_in, (3, 3), iterations=1)
    
    
    orig = cv2.bitwise_and(orig, orig, mask=cv2.add(frame_box_in, frame_box_out))
    
    contours, _ = cv2.findContours(frame_box_out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    corners = []
    for c in contours:
        if cv2.contourArea(c) < 800:
           continue


        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)

        box = np.array(box, dtype="int")

        
        box = perspective.order_points(box)
        box = np.array(box, dtype="int")
        (tl, tr, br, bl) = box

        corners.append(ObjectShape((tl, tr, br, bl)))
        cv2.drawContours(orig, [box], -1, (0,0,255), 2)
        
        
    contours, _ = cv2.findContours(frame_box_in, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    corners = []
    for c in contours:
        if cv2.contourArea(c) < 800:
           continue


        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)

        box = np.array(box, dtype="int")

        
        box = perspective.order_points(box)
        box = np.array(box, dtype="int")
        (tl, tr, br, bl) = box

        corners.append(ObjectShape((tl, tr, br, bl)))
        cv2.drawContours(orig, [box], -1, (255,0,0), 2)

    
    cv2.imshow(window_detection_name, orig)
    vid_writer.write(orig)

    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:
        break
        
cap.release()
vid_writer.release()
cv2.destroyAllWindows()
