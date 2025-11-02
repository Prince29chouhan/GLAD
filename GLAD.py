"""
Created on 2023/1/1
@Author: Guo Hanqing
@Email : guohanqing@westlake.edu.cn
@File : GLAD.py
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from detector_factory import (
    DetectorConfigurationError,
    available_backends,
    build_detectors,
)
from MOD2 import MOD2_global
from MOD2 import MOD2_local

from Functions import enlarge_region2


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the GLAD detector pipeline.")
    parser.add_argument(
        "--backend",
        choices=available_backends(),
        default="auto",
        help="Detection backend to use (defaults to TensorRT when available).",
    )
    parser.add_argument(
        "--weights-root",
        default="weights",
        help="Directory containing detector weight files (engines or .pt models).",
    )
    parser.add_argument(
        "--global-model",
        default="yolov5s_GLAD",
        help="Model stem for the global detector weights.",
    )
    parser.add_argument(
        "--local-model",
        default="yolov5s_GLAD-crop",
        help="Model stem for the local detector weights.",
    )
    parser.add_argument(
        "--video",
        default="phantom09",
        help="Video name (without extension) to process.",
    )
    parser.add_argument(
        "--video-root",
        default="/home/user-guo/data/ARD-MAV/videos",
        help="Directory containing the input videos.",
    )
    return parser.parse_args()


args = _parse_args()

try:
    detectors = build_detectors(
        args.backend,
        global_model=args.global_model,
        local_model=args.local_model,
        weights_root=args.weights_root,
    )
except DetectorConfigurationError as error:
    raise SystemExit(str(error)) from error

detector1, detector2, detector3, backend_name = detectors
print(f"Loaded {backend_name} detectors from {args.weights_root}")


sets_ordinary = ['phantom09', 'phantom10', 'phantom30', 'phantom47', 'phantom70']
sets_complex = ['phantom05', 'phantom08', 'phantom58', 'phantom65', 'phantom86']
sets_small = ['phantom19', 'phantom41', 'phantom43', 'phantom46', 'phantom63']

video_name = args.video
video_root = Path(args.video_root)
cap = cv2.VideoCapture(str(video_root / f"{video_name}.mp4"))

count = 0
flag = 0
prveframe = None
local_num = 0
a = 160

x2 = 0
y2 = 0
w2 = 0
h2 = 0

border = 1
border1 = 1
output_boxes = []


while cap.isOpened():
    ret, frame = cap.read()
    count = count + 1
    if not ret:
        break
    # frame = cv2.resize(frame, (1920, 1080), dst=None, interpolation=cv2.INTER_CUBIC)
    # print(ret)

    if prveframe is None:
        print('first frame input')
        prveframe = frame
        continue

    frame_show = frame.copy()
    width = frame.shape[1]
    height = frame.shape[0]
    t1 = time.time()

    if flag == 0:
        boxes = detector1.detect(frame)
        if len(boxes) == 0:
            boxes_MOD = MOD2_global(prveframe, frame)
            # print(boxes_MOD)
            if len(boxes_MOD) != 0:
                (x, y) = (boxes_MOD[0], boxes_MOD[1])
                (w, h) = (boxes_MOD[2], boxes_MOD[3])
                # init_rect = [x, y, w, h]
                # output_box = [count, x, y, w, h]
                # output_boxes.append(output_box)
                # # Draw bounding box and label
                # color = (255, 0, 0)
                # cv2.rectangle(frame_show, (x - border1, y - border1), (x + w + border1, y + h + border1), color, border, lineType=cv2.LINE_AA)
                # cv2.putText(frame_show, "Global MOD Detection Success", (600, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)
                # cv2.putText(frame_show, "Frame: {}".format(count), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # cv2.imshow('Detection', frame_show)
                # print('MOD Initialize Success:', init_rect)
                # flag = 1
                # local_num = 0
                x1, y1, w1, h1 = enlarge_region2(x, y, a, width, height)
                x2 = x - x1
                y2 = y - y1
                w2 = w
                h2 = h

                x_center = x2 + w2 / 2
                y_center = y2 + h2 / 2

                local_region = frame[y1:y1 + h1, x1:x1 + w1, :]
                boxes_local_yolo = detector3.detect(local_region, x_center, y_center)
                if len(boxes_local_yolo) != 0:
                    (x2, y2) = (boxes_local_yolo[0], boxes_local_yolo[1])
                    (w2, h2) = (boxes_local_yolo[2], boxes_local_yolo[3])
                    init_rect = [x2 + x1, y2 + y1, w2, h2]

                    output_box = [count, x2 + x1, y2 + y1, w2, h2]
                    output_boxes.append(output_box)
                    # Draw bounding box and label
                    color = (255, 0, 0)
                    cv2.rectangle(frame_show, (x2 + x1 - border1, y2 + y1 - border1), (x2 + x1 + w2 + border1, y2 + y1 + h2 + border1), color,
                                  border, lineType=cv2.LINE_AA)
                    cv2.putText(frame_show, "Global MOD Detection Success", (600, 100), cv2.FONT_HERSHEY_SIMPLEX, 2,
                                color, 2)

                    xleft = x2 + x1
                    ytop = y2 + y1
                    xright = x2 + x1 + w2
                    ybottom = y2 + y1 + h2

                    flag = 1
                    local_num = 0

                    x1, y1, w1, h1 = enlarge_region2(xleft, ytop, a, width, height)
                    x2 = init_rect[0] - x1
                    y2 = init_rect[1] - y1
                    status = 'Global MOD'
                else:
                    cv2.putText(frame_show, "Global YOLO and MOD Failed", (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                    flag = 0
                    init_rect = []
                    status = 'Both Failure'
            else:
                cv2.putText(frame_show, "Global YOLO and MOD Failed", (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                flag = 0
                init_rect = []
                status = 'Both Failure'
        else:
            (x, y) = (boxes[0], boxes[1])
            (w, h) = (boxes[2], boxes[3])
            init_rect = [x, y, w, h]
            output_box = [count, x, y, w, h]
            output_boxes.append(output_box)
            color = (0, 255, 255)
            cv2.putText(frame_show, "Global YOLO Detection Success", (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)
            cv2.rectangle(frame_show, (x - border1, y - border1), (x + w + border1, y + h + border1), color, border, lineType=cv2.LINE_AA)
            local_num = 0
            flag = 1
            x1, y1, w1, h1 = enlarge_region2(x, y, a, width, height)
            x2 = x - x1
            y2 = y - y1
            w2 = w
            h2 = h
            status = 'Global YOLO'
    else:
        track_crop1 = prveframe[y1:y1 + h1, x1:x1 + w1, :]
        track_crop2 = frame[y1:y1 + h1, x1:x1 + w1, :]
        x_prev = x2 + w2 / 2
        y_prev = y2 + h2 / 2
        boxes = detector2.detect(track_crop2, x_prev, y_prev)
        if len(boxes) == 0:
            boxes_MOD = MOD2_local(track_crop1, track_crop2, x_prev, y_prev)
            if len(boxes_MOD) != 0:
                (x2, y2) = (boxes_MOD[0], boxes_MOD[1])
                (w2, h2) = (boxes_MOD[2], boxes_MOD[3])
                init_rect = [x2 + x1, y2 + y1, w2, h2]
                output_box = [count, x2 + x1, y2 + y1, w2, h2]
                output_boxes.append(output_box)
                # Draw bounding box and label
                color = (255, 0, 0)
                cv2.rectangle(frame_show, (x2 + x1 - border1, y2 + y1 - border1), (x2 + x1 + w2 + border1, y2 + y1 + h2 + border1), color, border, lineType=cv2.LINE_AA)
                cv2.putText(frame_show, "Local MOD Success", (600, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)
                # cv2.rectangle(frame_show, (x1, y1), (x1 + w1, y1 + h1), (255, 255, 255), 2, lineType=cv2.LINE_AA)
                # cv2.putText(frame_show, "search region", (x1 + 20, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                local_num = 0
                flag = 1
                x1, y1, w1, h1 = enlarge_region2(x2 + x1, y2 + y1, a, width, height)
                status = 'Local MOD'
            else:
                cv2.putText(frame_show, "Local YOLO and MOD Failed", (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                init_rect = []
                status = 'Local Both Failure'
                local_num = local_num + 1
        else:
            (x2, y2) = (boxes[0], boxes[1])
            (w2, h2) = (boxes[2], boxes[3])
            init_rect = [x2 + x1, y2 + y1, w2, h2]
            output_box = [count, x2 + x1, y2 + y1, w2, h2]
            output_boxes.append(output_box)
            color = (0, 255, 255)
            cv2.rectangle(frame_show, (x2 + x1 - border1, y2 + y1 - border1), (x2 + x1 + w2 + border1, y2 + y1 + h2 + border1), color, border, lineType=cv2.LINE_AA)
            cv2.putText(frame_show, "Local YOLO Detection Success", (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)
            # cv2.rectangle(frame_show, (x1, y1), (x1 + w1, y1 + h1), (255, 255, 255), 2, lineType=cv2.LINE_AA)
            # cv2.putText(frame_show, "search region", (x1 + 20, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            local_num = 0
            flag = 1
            x1, y1, w1, h1 = enlarge_region2(x2 + x1, y2 + y1, a, width, height)
            status = 'Local YOLO'

        if local_num == 30:
            print('turn to global detection')
            color0 = (0, 0, 255)
            # cv2.putText(frame_show, "turn to global detection", (500, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, color0, 2)
            flag = 0

    fps = (1. / (time.time() - t1))
    cv2.putText(frame_show, "Frame: {}".format(count), (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # cv2.putText(frame_show, 'FPS: {:.2f}'.format(fps), (50, 30), 0, 1, (0, 0, 255), 2)
    # cv2.putText(frame_show, box_status, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
    color2 = (0, 0, 255)
    cv2.imshow('Detection', frame_show)
    # cv2.imwrite('./output/' + video_name + '/' + str(count) + '.jpg', frame_show)
    print('frame count: %d fps: %.2f' % (count, fps), end=' ')
    print(status, end=' ')
    print('bbox:', init_rect)
    prveframe = frame
    key = cv2.waitKey(10) & 0xff

    if key == 27 or key == ord('q'):
        break

# tracked_bb = np.array(output_boxes).astype(int)
# bbox_file = './output/' + video_name + '.txt'
# np.savetxt(bbox_file, tracked_bb, delimiter='\t', fmt='%d')
cap.release()
cv2.destroyAllWindows()



