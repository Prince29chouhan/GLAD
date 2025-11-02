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
from Functions import frame_stablize
from Functions import enlarge_region2


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the GLAD pipeline with motion compensation."
    )
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
        default="yolov5s_DT-Drone2",
        help="Model stem for the global detector weights.",
    )
    parser.add_argument(
        "--local-model",
        default="yolov5s_DT-Drone2-crop",
        help="Model stem for the local detector weights.",
    )
    parser.add_argument(
        "--video-root",
        default="/home/user-guo/data/ARD-MAV/videos",
        help="Directory containing the input videos.",
    )
    parser.add_argument(
        "--videos",
        nargs="+",
        default=["phantom09"],
        help="List of video names (without extension) to process.",
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
except DetectorConfigurationError as first_error:
    if args.global_model == "yolov5s_DT-Drone2" and args.local_model == "yolov5s_DT-Drone2-crop":
        fallback_global = "yolov5s_GLAD"
        fallback_local = "yolov5s_GLAD-crop"
        print(
            "Falling back to default GLAD weights because the DT-Drone2 weights"
            f" are unavailable: {first_error}"
        )
        detectors = build_detectors(
            args.backend,
            global_model=fallback_global,
            local_model=fallback_local,
            weights_root=args.weights_root,
        )
    else:
        raise SystemExit(str(first_error)) from first_error

detector1, detector2, _, backend_name = detectors
print(f"Loaded {backend_name} detectors from {args.weights_root}")

sets_ordinary = ['phantom09', 'phantom10', 'phantom30', 'phantom47', 'phantom70']
sets_complex = ['phantom05', 'phantom08', 'phantom58', 'phantom65', 'phantom86']
sets_small = ['phantom19', 'phantom41', 'phantom43', 'phantom46', 'phantom63']

sets_test = args.videos
video_root = Path(args.video_root)

border = 1

for i in range(len(sets_test)):
    video_name = sets_test[i]

    cap = cv2.VideoCapture(str(video_root / f"{video_name}.mp4"))

    count = 0
    flag = 0
    prveframe = None
    fail_num = 0
    a = 160

    print('read file: ', video_name)

    while cap.isOpened():
        ret, frame = cap.read()
        # print(ret)
        if not ret:
            break

        if prveframe is None:
            print('first frame input')
            prveframe = frame
            count = count + 1
            continue

        frame_show = frame.copy()
        width = frame.shape[1]
        height = frame.shape[0]
        t1 = time.time()

        if flag == 0:
            boxes = detector1.detect(frame)
            if len(boxes) == 0:
                boxes_MOD = MOD2_global(prveframe, frame)
                if len(boxes_MOD) != 0:
                    (x, y) = (boxes_MOD[0], boxes_MOD[1])
                    (w, h) = (boxes_MOD[2], boxes_MOD[3])

                    init_rect = [x, y, w, h]
                    flag = 1
                    fail_num = 0

                    xleft = x
                    ytop = y
                    xright = x + w
                    ybottom = y + h

                    x1, y1, w1, h1 = enlarge_region2(x, y, a, width, height)

                    # Draw the bounding box and label
                    color = (255, 0, 0)
                    cv2.rectangle(frame_show, (xleft, ytop), (xright, ybottom), color, border, lineType=cv2.LINE_AA)

                    x2 = x - x1
                    y2 = y - y1
                    w2 = w
                    h2 = h
                else:
                    flag = 0
                    init_rect = []
                    # status = 'Both Failure'
            else:
                (x, y) = (boxes[0], boxes[1])
                (w, h) = (boxes[2], boxes[3])
                init_rect = [x, y, w, h]

                xleft = x
                ytop = y
                xright = x + w
                ybottom = y + h

                color = (0, 255, 255)
                cv2.rectangle(frame_show, (xleft, ytop), (xright, ybottom), color, border, lineType=cv2.LINE_AA)
                fail_num = 0
                flag = 1
                x1, y1, w1, h1 = enlarge_region2(xleft, ytop, a, width, height)

                x2 = x - x1
                y2 = y - y1
                w2 = w
                h2 = h
                # status = 'Global YOLO'
        else:
            homo_inv = frame_stablize(prveframe, frame)
            search_box = np.array([[x1, y1], [x1 + w1, y1], [x1 + w1, y1 + h1], [x1, y1 + h1]], dtype=np.float32).reshape(-1, 1, 2)
            search_box_s = cv2.perspectiveTransform(search_box, homo_inv)
            search_box_new = np.array(search_box_s, dtype=np.int32).reshape(4, 2)

            detect_box = np.array([[xleft, ytop], [xright, ytop], [xright, ybottom], [xleft, ybottom]], dtype=np.float32).reshape(-1, 1, 2)
            detect_box_s = cv2.perspectiveTransform(detect_box, homo_inv)
            detect_box_new = np.array(detect_box_s, dtype=np.int32).reshape(4, 2)

            if search_box_new[0][0] < 0:
                search_box_new[0][0] = 0

            if search_box_new[0][1] < 0:
                search_box_new[0][1] = 0

            if search_box_new[2][1] > height:
                search_box_new[2][1] = height

            if search_box_new[1][0] > width:
                search_box_new[1][0] = width

            track_crop1 = prveframe[search_box_new[0][1]:search_box_new[2][1], search_box_new[0][0]:search_box_new[1][0], :]
            track_crop2 = frame[search_box_new[0][1]:search_box_new[2][1], search_box_new[0][0]:search_box_new[1][0], :]
            x_prve = (detect_box_new[0][0] + detect_box_new[2][0] - search_box_new[0][0] * 2) / 2
            y_prve = (detect_box_new[0][1] + detect_box_new[2][1] - search_box_new[0][1] * 2) / 2
            boxes = detector2.detect(track_crop2, x_prve, y_prve)

            if len(boxes) == 0:
                boxes_MOD = MOD2_local(track_crop1, track_crop2, x_prve, y_prve)
                if len(boxes_MOD) != 0:
                    (x2, y2) = (boxes_MOD[0], boxes_MOD[1])
                    (w2, h2) = (boxes_MOD[2], boxes_MOD[3])

                    init_rect = [x2 + search_box_new[0][0], y2 + search_box_new[0][1], w2, h2]
                    xleft = x2 + search_box_new[0][0]
                    ytop = y2 + search_box_new[0][1]
                    xright = x2 + search_box_new[0][0] + w2
                    ybottom = y2 + search_box_new[0][1] + h2

                    # Draw the bounding box and label
                    color = (255, 0, 0)
                    cv2.rectangle(frame_show, (xleft, ytop), (xright, ybottom), color, border, lineType=cv2.LINE_AA)
                    cv2.rectangle(frame_show, (search_box_new[0][0], search_box_new[0][1]), (search_box_new[1][0], search_box_new[2][1]), (255, 255, 255), 2, lineType=cv2.LINE_AA)
                    cv2.putText(frame_show, "search region", (search_box_new[0][0] + 20, search_box_new[0][1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    fail_num = 0
                    flag = 2
                    x1, y1, w1, h1 = enlarge_region2(xleft, ytop, a, width, height)
                    # status = 'Local MOD'
                else:
                    fail_num = fail_num + 1
                    init_rect = []
                    # status = 'Local Both Failure'
            else:
                (x2, y2) = (boxes[0], boxes[1])
                (w2, h2) = (boxes[2], boxes[3])

                init_rect = [x2 + search_box_new[0][0], y2 + search_box_new[0][1], w2, h2]
                xleft = x2 + search_box_new[0][0]
                ytop = y2 + search_box_new[0][1]
                xright = x2 + search_box_new[0][0] + w2
                ybottom = y2 + search_box_new[0][1] + h2

                color = (0, 255, 255)
                cv2.rectangle(frame_show, (xleft, ytop), (xright, ybottom), color, border, lineType=cv2.LINE_AA)
                cv2.rectangle(frame_show, (search_box_new[0][0], search_box_new[0][1]), (search_box_new[1][0], search_box_new[2][1]), (255, 255, 255), 2, lineType=cv2.LINE_AA)
                cv2.putText(frame_show, "search region", (search_box_new[0][0] + 20, search_box_new[0][1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                fail_num = 0
                flag = 2
                x1, y1, w1, h1 = enlarge_region2(xleft, ytop, a, width, height)
            if fail_num == 30:
                print('turn to global re-detection')
                flag = 0

        print(video_name, end=" ")
        print('frame count: %d' % count, end=' ')
        print('bbox:', init_rect)
        cv2.imshow('GLAD', frame_show)
        count = count + 1
        prveframe = frame
        key = cv2.waitKey(10) & 0xff

        if key == 27 or key == ord('q'):
            break
    cap.release()

cv2.destroyAllWindows()




