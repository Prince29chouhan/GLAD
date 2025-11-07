import cv2
import numpy as np
from torch import nn
from torchvision import transforms
import torch.nn.functional as F
import torch

import xml.etree.ElementTree as ET


def motion_compensate(frame1, frame2):
    # grid-based KLT tracking
    lk_params = dict(winSize=(15, 15), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    # Create randomly generated colors
    # color = np.random.randint(0, 255, (3000, 3))

    width = frame2.shape[1]
    height = frame2.shape[0]
    gridSizeW = 32 * 3
    gridSizeH = 24 * 3
    p1 = []
    grid_numW = int(width / gridSizeW - 1)
    grid_numH = int(height / gridSizeH - 1)
    for i in range(grid_numW):
        for j in range(grid_numH):
            point = (np.float32(i * gridSizeW + gridSizeW / 2.0), np.float32(j * gridSizeH + gridSizeH / 2.0))
            p1.append(point)

    p1 = np.array(p1)
    pts_num = grid_numW * grid_numH
    pts_prev = p1.reshape(pts_num, 1, 2)

    pts_cur, st, err = cv2.calcOpticalFlowPyrLK(frame1, frame2, pts_prev, None, **lk_params)

    # Select good points
    good_new = pts_cur[st == 1]  # Tracked points in the current frame
    good_old = pts_prev[st == 1]  # Tracked points in the previous frame

    points_new = []
    points_old = []
    motion_distance = []
    # Draw tracking boxes
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        motion_distance0 = np.sqrt((a - c) * (a - c) + (b - d) * (b - d))
        if motion_distance0 > 50:
            continue

        point_new = np.array([a, b])
        point_old = np.array([c, d])
        points_new.append(point_new)
        points_old.append(point_old)

        motion_distance.append(motion_distance0)

    motion_dist = np.array(motion_distance)
    avg_dist = np.mean(motion_dist)

    if len(good_old) < 9:
        homography_matrix = np.array([[0.999, 0, 0], [0, 0.999, 0], [0, 0, 1]])
    else:
        homography_matrix, status = cv2.findHomography(good_new, good_old, cv2.RANSAC, 3.0)

    # homography_matrix, status = cv2.findHomography(good_new, good_old, cv2.RANSAC, 3.0)
    # print('homography matrix:', homography_matrix)

    # Warp the image using the transformation matrix
    compensated = cv2.warpPerspective(frame1, homography_matrix, (width, height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    # Compute the mask
    vertex = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32).reshape(-1, 1, 2)
    homo_inv = np.linalg.inv(homography_matrix)
    vertex_trans = cv2.perspectiveTransform(vertex, homo_inv)
    vertex_transformed = np.array(vertex_trans, dtype=np.int32).reshape(1, 4, 2)
    im = np.zeros(frame1.shape[:2], dtype='uint8')
    cv2.polylines(im, vertex_transformed, 1, 255)
    cv2.fillPoly(im, vertex_transformed, 255)
    mask = 255 - im

    return compensated, mask, avg_dist


def motion_compensate_local(frame1, frame2):
    # grid-based KLT tracking
    lk_params = dict(winSize=(15, 15), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03))

    # Create randomly generated colors
    # color = np.random.randint(0, 255, (3000, 3))

    width = frame2.shape[1]
    height = frame2.shape[0]
    gridSizeW = 8 * 1.5
    gridSizeH = 8 * 1.5
    p1 = []
    grid_numW = int(width / gridSizeW - 1)
    grid_numH = int(height / gridSizeH - 1)
    for i in range(grid_numW):
        for j in range(grid_numH):
            point = (np.float32(i * gridSizeW + gridSizeW / 2.0), np.float32(j * gridSizeH + gridSizeH / 2.0))
            p1.append(point)

    p1 = np.array(p1)
    pts_num = grid_numW * grid_numH
    pts_prev = p1.reshape(pts_num, 1, 2)

    pts_cur, st, err = cv2.calcOpticalFlowPyrLK(frame1, frame2, pts_prev, None, **lk_params)

    # Select good points
    good_new = pts_cur[st == 1]  # Tracked points in the current frame
    good_old = pts_prev[st == 1]  # Tracked points in the previous frame
    # print('local points num:', len(good_old))
    if len(good_old) < 18:
        homography_matrix = np.array([[0.999, 0, 0], [0, 0.999, 0], [0, 0, 1]])
    else:
        homography_matrix, status = cv2.findHomography(good_new, good_old, cv2.RANSAC, 3.0)

    # Warp the image using the transformation matrix
    compensated = cv2.warpPerspective(frame1, homography_matrix, (width, height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    # Compute the mask
    vertex = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32).reshape(-1, 1, 2)
    homo_inv = np.linalg.inv(homography_matrix)
    vertex_trans = cv2.perspectiveTransform(vertex, homo_inv)
    vertex_transformed = np.array(vertex_trans, dtype=np.int32).reshape(1, 4, 2)
    im = np.zeros(frame1.shape[:2], dtype='uint8')
    cv2.polylines(im, vertex_transformed, 1, 255)
    cv2.fillPoly(im, vertex_transformed, 255)
    mask = 255 - im

    return compensated, mask, homo_inv


def frame_stablize(frame1, frame2):
    # grid-based KLT tracking
    blur_kernel = 11
    prevFrame = cv2.GaussianBlur(frame1, (blur_kernel, blur_kernel), 0)  # Gaussian blur for denoising
    prevFrame = cv2.cvtColor(prevFrame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    currentFame = cv2.GaussianBlur(frame2, (blur_kernel, blur_kernel), 0)
    currentFrame = cv2.cvtColor(currentFame, cv2.COLOR_BGR2GRAY)

    lk_params = dict(winSize=(15, 15), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    width = frame2.shape[1]
    height = frame2.shape[0]
    gridSizeW = 32
    gridSizeH = 24
    p1 = []
    grid_numW = int(width / gridSizeW - 1)
    grid_numH = int(height / gridSizeH - 1)
    for i in range(grid_numW):
        for j in range(grid_numH):
            point = (np.float32(i * gridSizeW + gridSizeW / 2.0), np.float32(j * gridSizeH + gridSizeH / 2.0))
            p1.append(point)

    p1 = np.array(p1)
    pts_num = grid_numW * grid_numH
    pts_prev = p1.reshape(pts_num, 1, 2)

    pts_cur, st, err = cv2.calcOpticalFlowPyrLK(prevFrame, currentFrame, pts_prev, None, **lk_params)

    # Select good points
    good_new = pts_cur[st == 1]  # Tracked points in the current frame
    good_old = pts_prev[st == 1]  # Tracked points in the previous frame

    points_new = []
    points_old = []
    # Draw tracking boxes
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        motion_distance0 = np.sqrt((a - c) * (a - c) + (b - d) * (b - d))
        if motion_distance0 > 50:
            continue

        point_new = np.array([a, b])
        point_old = np.array([c, d])
        points_new.append(point_new)
        points_old.append(point_old)

    points_new = np.array(points_new)
    points_old = np.array(points_old)

    # Warp the image using the perspective transformation matrix
    homography_matrix, status = cv2.findHomography(points_new, points_old, cv2.RANSAC, 3.0)
    img_compensate = cv2.warpPerspective(frame2, homography_matrix, (width, height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    homo_inv = np.linalg.inv(homography_matrix)
    # # Stabilize the image using the affine transformation matrix
    # # Find affine transformation matrix
    # m, _ = cv2.estimateAffinePartial2D(points_new, points_old, maxIters=200, ransacReprojThreshold=3)
    #
    # # Extract translation
    # dx = m[0, 2]
    # dy = m[1, 2]
    #
    # # Extract rotation angle
    # da = np.arctan2(m[1, 0], m[0, 0])
    #
    # # Store transformation
    # m = np.zeros((2, 3), np.float32)
    # m[0, 0] = np.cos(da)
    # m[0, 1] = -np.sin(da)
    # m[1, 0] = np.sin(da)
    # m[1, 1] = np.cos(da)
    # m[0, 2] = dx
    # m[1, 2] = dy
    #
    # # Warp the image using the transformation matrix
    # img_compensate = cv2.warpAffine(frame2, m, (width, height))
    # m_inv = np.linalg.inv(m)

    return homo_inv


def enlargebox(x, y, w, h, a, width, height):
    # xa = int(w * a)
    # ya = int(h * a)
    # if xa > 10:
    #     xa = 10
    #
    # if ya > 10:
    #     ya = 10

    xa = a
    ya = a

    x1 = x - xa
    y1 = y - ya
    w1 = w + xa * 2
    h1 = h + ya * 2

    if x1 < 0:
        x1 = 0

    if y1 < 0:
        y1 = 0

    if x1 + w1 >= width:
        w1 = width - x1 - 1

    if y1 + h1 >= height:
        h1 = height - y1 - 1

    return int(x1), int(y1), int(w1), int(h1)


def enlarge_region(x, y, w, h, a, width, height):
    x1 = x - a
    y1 = y - a
    w1 = w + a * 2
    h1 = h + a * 2

    if x1 < 0:
        x1 = 0

    if y1 < 0:
        y1 = 0

    if x1 + w1 >= width:
        w1 = width - x1 - 1

    if y1 + h1 >= height:
        h1 = height - y1 - 1

    return int(x1), int(y1), int(w1), int(h1)


def enlarge_region2(x, y, a, width, height):
    x1 = x - a
    y1 = y - a
    w1 = a * 2
    h1 = a * 2

    if x1 < 0:
        x1 = 0

    if y1 < 0:
        y1 = 0

    if x1 + w1 >= width:
        x1 = width - w1

    if y1 + h1 >= height:
        y1 = height - h1

    return int(x1), int(y1), int(w1), int(h1)


def cal_iou(box1, box2):
    """

    :param box1: xywh (top-left corner with width and height)
    :param box2: xywh
    :transfer to xyxy
    """
    x1min, y1min, x1max, y1max = box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]
    x2min, y2min, x2max, y2max = box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]
    # Compute the area of the two boxes
    s1 = (y1max - y1min + 1.) * (x1max - x1min + 1.)
    s2 = (y2max - y2min + 1.) * (x2max - x2min + 1.)

    # Compute the coordinates of the intersection
    xmin = max(x1min, x2min)
    ymin = max(y1min, y2min)
    xmax = min(x1max, x2max)
    ymax = min(y1max, y2max)

    inter_h = max(ymax - ymin + 1, 0)
    inter_w = max(xmax - xmin + 1, 0)

    intersection = inter_h * inter_w
    union = s1 + s2 - intersection

    # Compute IoU
    iou = intersection / union
    return iou


def cal_center_distance(box1, box2):
    """
    Calculate the distance between the centers of two boxes.
    :param box1: xyxy (top-left and bottom-right)
    :param box2: xyxy
    :return:
    """
    center1 = ((box1[0] + box1[2]) // 2, (box1[1] + box1[3]) // 2)
    center2 = ((box2[0] + box2[2]) // 2, (box2[1] + box2[3]) // 2)
    dis = ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5

    return dis


def dist(x1, y1, x2, y2):
    distance = np.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))
    return distance


def rect_dist(x1, y1, w1, h1, x2, y2, w2, h2):
    # Convert to top-left and bottom-right coordinates
    x1b = x1 + w1
    y1b = y1 + h1
    x2b = x2 + w2
    y2b = y2 + h2

    left = x2b < x1
    right = x1b < x2
    bottom = y2b < y1
    top = y1b < y2

    if top and left:
        return dist(x1, y1b, x2b, y2)
    elif left and bottom:
        return dist(x1, y1, x2b, y2b)
    elif bottom and right:
        return dist(x1b, y1, x2, y2b)
    elif right and top:
        return dist(x1b, y1b, x2, y2)
    elif left:
        return x1 - x2b
    elif right:
        return x2 - x1b
    elif bottom:
        return y1 - y2b
    elif top:
        return y2 - y1b
    else:  # rectangles intersect
        return 0


def two2one(x1, y1, w1, h1, x2, y2, w2, h2):
    """
    Merge two rectangles into a larger rectangle.
    Input: two rectangles represented by top-left corner plus width and height.
    Return: the merged rectangle as top-left and bottom-right coordinates.
    """
    # Convert to top-left and bottom-right coordinates
    x1b = x1 + w1
    y1b = y1 + h1
    x2b = x2 + w2
    y2b = y2 + h2

    x = min(x1, x2)
    y = min(y1, y2)
    xb = max(x1b, x2b)
    yb = max(y1b, y2b)

    return x, y, xb, yb


def box_select(boxes1):

    """
    Merge nearby boxes and keep either the merged boxes or the untouched ones.
    Input: a list of boxes, e.g., [[12, 23, 45, 56], [36, 25, 45, 63], [30, 25, 60, 35]].
    Return: the new boxes; merged boxes are set to [], so the result may look like [[], [], [50, 23, 65, 50]].
    """

    # print("boxes1:", boxes1)
    if len(boxes1) > 0:
        for bi in range(len(boxes1)):
            for bj in range(len(boxes1)):
                if bi != bj:
                    if len(boxes1[bi]) == 4 and len(boxes1[bj]) == 4:
                        x1, y1, w1, h1 = int(boxes1[bi][0]), int(boxes1[bi][1]), int(boxes1[bi][2]), int(boxes1[bi][3])
                        x2, y2, w2, h2 = int(boxes1[bj][0]), int(boxes1[bj][1]), int(boxes1[bj][2]), int(boxes1[bj][3])

                        dis = rect_dist(x1, y1, w1, h1, x2, y2, w2, h2)
                        if dis < 15:
                            # print('merge boxes')
                            x, y, xb, yb = two2one(x1, y1, w1, h1, x2, y2, w2, h2)
                            boxes1[bj][0] = x
                            boxes1[bj][1] = y
                            boxes1[bj][2] = xb - x
                            boxes1[bj][3] = yb - y
                            boxes1[bi] = np.zeros(4)

    return boxes1


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MyNet(nn.Module):

    def __init__(self, num_classes=2) -> None:
        super(MyNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.model(x)
        return x


def Mynet_infer(src):
    data_transform = transforms.Compose([transforms.ToTensor()])
    size = 32
    img = cv2.resize(src, (size, size))
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    model = Net()
    # load model weights
    model_weight_path = "./weights/Net_best.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=torch.device('cpu')))
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img))
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
        # print(predict_cla)

    return predict_cla


def readGTbox(xml_file):
    global x3, y3, w3, h3, GT_box
    tree = ET.parse(xml_file)
    root = tree.getroot()

    if root.find('object') == None:
        GT_box = []
        return GT_box
    else:
        for obj in root.iter('object'):
            xmlbox = obj.find('bndbox')
            b = [int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)),
                 int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text))]

            x3 = b[0]
            y3 = b[1]
            w3 = b[2] - b[0]
            h3 = b[3] - b[1]

            GT_box = np.array([x3, y3, w3, h3])

        return GT_box



