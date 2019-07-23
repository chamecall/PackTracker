from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time


def convert_back(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cv_draw_boxes(detections, img):
    for detection in detections:
        x, y, w, h = detection[2][0], \
                     detection[2][1], \
                     detection[2][2], \
                     detection[2][3]
        xmin, ymin, xmax, ymax = convert_back(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 0, 255), 1)
        cv2.putText(img,
                    detection[0].decode() +
                    " [" + str(round(detection[1] * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
    return img


class Yolo:
    def __init__(self):
        self.netMain = None
        self.metaMain = None
        self.altNames = None

        configPath = "/home/algernon/samba/video_queue/omega-packaging/experiments/exp-001-Simple-objects-detection/config/omega-pack-yolov3.cfg"
        weightPath = "/home/algernon/samba/video_queue/omega-packaging/experiments/exp-001-Simple-objects-detection/models/omega-pack-yolov3.saved_backup"
        metaPath = "/home/algernon/samba/video_queue/omega-packaging/experiments/exp-001-Simple-objects-detection/config/omega-pack.data"
        if not os.path.exists(configPath):
            raise ValueError("Invalid config path `" +
                             os.path.abspath(configPath) + "`")
        if not os.path.exists(weightPath):
            raise ValueError("Invalid weight path `" +
                             os.path.abspath(weightPath) + "`")
        if not os.path.exists(metaPath):
            raise ValueError("Invalid data file path `" +
                             os.path.abspath(metaPath) + "`")
        if self.netMain is None:
            self.netMain = Darknet.load_net_custom(configPath.encode(
                "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
        if self.metaMain is None:
            self.metaMain = Darknet.load_meta(metaPath.encode("ascii"))
        if self.altNames is None:
            try:
                with open(metaPath) as metaFH:
                    metaContents = metaFH.read()
                    import re
                    match = re.search("names *= *(.*)$", metaContents,
                                      re.IGNORECASE | re.MULTILINE)
                    if match:
                        result = match.group(1)
                    else:
                        result = None
                    try:
                        if os.path.exists(result):
                            with open(result) as namesFH:
                                namesList = namesFH.read().strip().split("\n")
                                self.altNames = [x.strip() for x in namesList]
                    except TypeError:
                        pass
            except Exception:
                pass

        self.darknet_image = Darknet.make_image(Darknet.network_width(self.netMain),
                                                Darknet.network_height(self.netMain), 3)

    def forward(self, image):
        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                   (Darknet.network_width(self.netMain),
                                    Darknet.network_height(self.netMain)),
                                   interpolation=cv2.INTER_LINEAR)

        Darknet.copy_image_from_bytes(self.darknet_image, frame_resized.tobytes())
        detections = Darknet.detect_image(self.netMain, self.metaMain, self.darknet_image, thresh=0.25)
        image = cv_draw_boxes(detections, frame_resized)
        cv2.imshow('mini result', image)
        return detections

