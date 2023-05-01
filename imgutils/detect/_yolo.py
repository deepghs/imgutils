import math
from typing import List

import numpy as np
from PIL import Image


def _yolo_xywh2xyxy(x: np.ndarray) -> np.ndarray:
    """
    Copied from yolov8.

    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray) or (torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.
    Returns:
        y (np.ndarray) or (torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def _yolo_nms(boxes, scores, thresh: float = 0.7) -> List[int]:
    """
    dets: ndarray, (num_boxes, 5)
        每一行表示一个bounding box：[xmin, ymin, xmax, ymax, score]
        其中xmin, ymin, xmax, ymax分别表示框的左上角和右下角坐标，score表示框的分数
    thresh: float
        两个框的IoU阈值
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # 按照score降序排列
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        # 计算其他所有框与当前框的IoU
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)

        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        # 保留IoU小于阈值的框
        inds = np.where(iou <= thresh)[0]
        order = order[inds + 1]

    return keep


def _image_preprocess(image: Image.Image, max_infer_size: int = 1216, align: int = 32):
    old_width, old_height = image.width, image.height
    new_width, new_height = old_width, old_height
    r = max_infer_size / max(new_width, new_height)
    if r < 1:
        new_width, new_height = new_width * r, new_height * r
    new_width = int(math.ceil(new_width / align) * align)
    new_height = int(math.ceil(new_height / align) * align)
    image = image.resize((new_width, new_height))
    return image, (old_width, old_height), (new_width, new_height)


def _xy_postprocess(x, y, old_size, new_size):
    old_width, old_height = old_size
    new_width, new_height = new_size
    x, y = x / new_width * old_width, y / new_height * old_height
    x = int(np.clip(x, a_min=0, a_max=old_width).round())
    y = int(np.clip(y, a_min=0, a_max=old_height).round())
    return x, y


def _data_postprocess(output, conf_threshold, iou_threshold, old_size, new_size, labels: List[str]):
    max_scores = output[4:, :].max(axis=0)
    output = output[:, max_scores > conf_threshold].transpose(1, 0)
    boxes = output[:, :4]
    scores = output[:, 4:]
    filtered_max_scores = scores.max(axis=1)

    if not boxes.size:
        return []

    boxes = _yolo_xywh2xyxy(boxes)
    idx = _yolo_nms(boxes, filtered_max_scores, thresh=iou_threshold)
    boxes, scores = boxes[idx], scores[idx]

    detections = []
    for box, score in zip(boxes, scores):
        x0, y0 = _xy_postprocess(box[0], box[1], old_size, new_size)
        x1, y1 = _xy_postprocess(box[2], box[3], old_size, new_size)
        max_score_id = score.argmax()
        detections.append(((x0, y0, x1, y1), labels[max_score_id], float(score[max_score_id])))

    return detections
