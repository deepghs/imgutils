"""
This module provides utility functions for processing and post-processing image data, particularly for object detection tasks using YOLO-like models. It includes functions for bounding box coordinate conversion, non-maximum suppression (NMS), image preprocessing, and detection result post-processing.

The module contains helper functions that are commonly used in the pipeline of object detection models, from preparing input images to interpreting and refining the model's output.
"""

import math
from typing import List

import numpy as np
from PIL import Image


def _yolo_xywh2xyxy(x: np.ndarray) -> np.ndarray:
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format.

    This function is adapted from YOLOv8 and transforms the center-based representation
    to a corner-based representation of bounding boxes.

    :param x: Input bounding box coordinates in (x, y, width, height) format.
    :type x: np.ndarray

    :return: Bounding box coordinates in (x1, y1, x2, y2) format.
    :rtype: np.ndarray

    :Example:

    >>> import numpy as np
    >>> boxes = np.array([[10, 10, 20, 20]])
    >>> _yolo_xywh2xyxy(boxes)
    array([[  0.,   0.,  20.,  20.]])
    """
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def _yolo_nms(boxes, scores, thresh: float = 0.7) -> List[int]:
    """
    Perform Non-Maximum Suppression (NMS) on bounding boxes.

    This function applies NMS to remove overlapping bounding boxes, keeping only the most confident detections.

    :param boxes: Array of bounding boxes, each in the format [xmin, ymin, xmax, ymax].
    :type boxes: np.ndarray
    :param scores: Array of confidence scores for each bounding box.
    :type scores: np.ndarray
    :param thresh: IoU threshold for considering boxes as overlapping. Default is 0.7.
    :type thresh: float

    :return: List of indices of the boxes to keep after NMS.
    :rtype: List[int]

    :Example:

    >>> boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30]])
    >>> scores = np.array([0.9, 0.8, 0.7])
    >>> _yolo_nms(boxes, scores, 0.5)
    [0, 2]
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)

        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= thresh)[0]
        order = order[inds + 1]

    return keep


def _image_preprocess(image: Image.Image, max_infer_size: int = 1216, align: int = 32):
    """
    Preprocess an input image for inference.

    This function resizes the image while maintaining its aspect ratio, and ensures
    the dimensions are multiples of 'align'.

    :param image: Input image to be preprocessed.
    :type image: Image.Image
    :param max_infer_size: Maximum size (width or height) of the processed image. Default is 1216.
    :type max_infer_size: int
    :param align: Value to align the image dimensions to. Default is 32.
    :type align: int

    :return: A tuple containing:
        - The preprocessed image
        - Original image dimensions (width, height)
        - New image dimensions (width, height)
    :rtype: tuple(Image.Image, tuple(int, int), tuple(int, int))

    :Example:

    >>> from PIL import Image
    >>> img = Image.new('RGB', (1000, 800))
    >>> processed_img, old_size, new_size = _image_preprocess(img)
    >>> print(old_size, new_size)
    (1000, 800) (1216, 992)
    """
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
    """
    Convert coordinates from the preprocessed image size back to the original image size.

    :param x: X-coordinate in the preprocessed image.
    :type x: float
    :param y: Y-coordinate in the preprocessed image.
    :type y: float
    :param old_size: Original image dimensions (width, height).
    :type old_size: tuple(int, int)
    :param new_size: Preprocessed image dimensions (width, height).
    :type new_size: tuple(int, int)

    :return: Adjusted (x, y) coordinates for the original image size.
    :rtype: tuple(int, int)

    :Example:

    >>> _xy_postprocess(100, 100, (1000, 800), (1216, 992))
    (82, 80)
    """
    old_width, old_height = old_size
    new_width, new_height = new_size
    x, y = x / new_width * old_width, y / new_height * old_height
    x = int(np.clip(x, a_min=0, a_max=old_width).round())
    y = int(np.clip(y, a_min=0, a_max=old_height).round())
    return x, y


def _data_postprocess(output, conf_threshold, iou_threshold, old_size, new_size, labels: List[str]):
    """
    Post-process the raw output from the object detection model.

    This function applies confidence thresholding, non-maximum suppression, and
    converts the coordinates back to the original image size.

    :param output: Raw output from the object detection model.
    :type output: np.ndarray
    :param conf_threshold: Confidence threshold for filtering detections.
    :type conf_threshold: float
    :param iou_threshold: IoU threshold for non-maximum suppression.
    :type iou_threshold: float
    :param old_size: Original image dimensions (width, height).
    :type old_size: tuple(int, int)
    :param new_size: Preprocessed image dimensions (width, height).
    :type new_size: tuple(int, int)
    :param labels: List of class labels.
    :type labels: List[str]

    :return: List of detections, each in the format ((x0, y0, x1, y1), label, confidence).
    :rtype: List[tuple(tuple(int, int, int, int), str, float)]

    :Example:

    >>> output = np.array([[10, 10, 20, 20, 0.9, 0.1]])
    >>> _data_postprocess(output, 0.5, 0.5, (100, 100), (128, 128), ['cat', 'dog'])
    [((7, 7, 15, 15), 'cat', 0.9)]
    """
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
