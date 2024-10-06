"""
This module provides functions for calculating similarities between bounding boxes and detections.

It includes functions to calculate Intersection over Union (IoU) for individual bounding boxes,
compute similarities between lists of bounding boxes, and compare detections with labels.
The module is designed to work with various types of bounding box representations and
offers different modes for aggregating similarity scores.

Key components:

- calculate_iou: Computes IoU between two bounding boxes
- bboxes_similarity: Calculates similarities between two lists of bounding boxes
- detection_similarity: Compares two lists of detections, considering both bounding boxes and labels

This module is particularly useful for tasks involving object detection, 
image segmentation, and evaluation of detection algorithms.
"""

from typing import List, Literal, Union

import numpy as np

from .base import BBoxTyping, BBoxWithScoreAndLabel


def calculate_iou(box1: BBoxTyping, box2: BBoxTyping) -> float:
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.

    :param box1: The first bounding box, represented as (x1, y1, x2, y2).
    :type box1: BBoxTyping
    :param box2: The second bounding box, represented as (x1, y1, x2, y2).
    :type box2: BBoxTyping
    :return: The IoU value between the two bounding boxes.
    :rtype: float

    This function computes the IoU, which is a measure of the overlap between two bounding boxes.
    The IoU is calculated as the area of intersection divided by the area of union of the two boxes.

    Example::
        >>> box1 = (0, 0, 2, 2)
        >>> box2 = (1, 1, 3, 3)
        >>> iou = calculate_iou(box1, box2)
        >>> print(f"IoU: {iou:.4f}")
        IoU: 0.1429
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = intersection / (area1 + area2 - intersection + 1e-6)
    return float(iou)


def bboxes_similarity(bboxes1: List[BBoxTyping], bboxes2: List[BBoxTyping],
                      mode: Literal['max', 'mean', 'raw'] = 'mean') -> Union[float, List[float]]:
    """
    Calculate the similarity between two lists of bounding boxes.

    :param bboxes1: First list of bounding boxes.
    :type bboxes1: List[BBoxTyping]
    :param bboxes2: Second list of bounding boxes.
    :type bboxes2: List[BBoxTyping]
    :param mode: The mode for calculating similarity. Options are 'max', 'mean', or 'raw'. Defaults to 'mean'.
    :type mode: Literal['max', 'mean', 'raw']
    :return: The similarity score or list of scores, depending on the mode.
    :rtype: Union[float, List[float]]
    :raises ValueError: If the lengths of bboxes1 and bboxes2 do not match, or if an unknown mode is specified.

    This function computes the similarity between two lists of bounding boxes using the Hungarian algorithm
    to find the optimal assignment. It then returns the similarity based on the specified mode:

    - ``max``: Returns the maximum IoU among all matched pairs.
    - ``mean``: Returns the average IoU of all matched pairs.
    - ``raw``: Returns a list of IoU values for all matched pairs.

    Example::
        >>> bboxes1 = [(0, 0, 2, 2), (3, 3, 5, 5)]
        >>> bboxes2 = [(1, 1, 3, 3), (4, 4, 6, 6)]
        >>> similarity = bboxes_similarity(bboxes1, bboxes2, mode='mean')
        >>> print(f"Mean similarity: {similarity:.4f}")
        Mean similarity: 0.1429
    """
    if len(bboxes1) != len(bboxes2):
        raise ValueError(f'Length of bboxes lists not match - {len(bboxes1)} vs {len(bboxes2)}.')

    n = len(bboxes1)
    iou_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            iou_matrix[i, j] = calculate_iou(bboxes1[i], bboxes2[j])

    # import here for faster launching speed
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    similarities = iou_matrix[row_ind, col_ind]
    if mode == 'max':
        return float(similarities.max())
    elif mode == 'mean':
        return float(similarities.mean())
    elif mode == 'raw':
        return similarities.tolist()
    else:
        raise ValueError(f'Unknown similarity mode for bboxes - {mode!r}.')


def detection_similarity(detect1: List[BBoxWithScoreAndLabel], detect2: List[BBoxWithScoreAndLabel],
                         mode: Literal['max', 'mean', 'raw'] = 'mean') -> Union[float, List[float]]:
    """
    Calculate the similarity between two lists of detections, considering both bounding boxes and labels.

    :param detect1: First list of detections, each containing a bounding box, label, and score.
    :type detect1: List[BBoxWithScoreAndLabel]
    :param detect2: Second list of detections, each containing a bounding box, label, and score.
    :type detect2: List[BBoxWithScoreAndLabel]
    :param mode: The mode for calculating similarity. Options are 'max', 'mean', or 'raw'. Defaults to 'mean'.
    :type mode: Literal['max', 'mean', 'raw']
    :return: The similarity score or list of scores, depending on the mode.
    :rtype: Union[float, List[float]]
    :raises ValueError: If the number of bounding boxes for any label doesn't match between detect1 and detect2,
                        or if an unknown mode is specified.

    This function compares two lists of detections by:

    1. Grouping detections by their labels.
    2. For each label, calculating the similarity between the corresponding bounding boxes.
    3. Aggregating the similarities based on the specified mode.

    The function ensures that for each label, the number of bounding boxes matches between detect1 and detect2.

    Example::
        >>> detect1 = [((0, 0, 2, 2), 'car', 0.9), ((3, 3, 5, 5), 'person', 0.8)]
        >>> detect2 = [((1, 1, 3, 3), 'car', 0.85), ((4, 4, 6, 6), 'person', 0.75)]
        >>> similarity = detection_similarity(detect1, detect2, mode='mean')
        >>> print(f"Mean detection similarity: {similarity:.4f}")
        Mean detection similarity: 0.1429
    """
    labels = sorted({*(l for _, l, _ in detect1), *(l for _, l, _ in detect2)})
    sims = []
    for current_label in labels:
        bboxes1 = [bbox for bbox, label, _ in detect1 if label == current_label]
        bboxes2 = [bbox for bbox, label, _ in detect2 if label == current_label]

        if len(bboxes1) != len(bboxes2):
            raise ValueError(f'Length of bboxes not match on label {current_label!r}'
                             f' - {len(bboxes1)} vs {len(bboxes2)}.')

        sims.extend(bboxes_similarity(
            bboxes1=bboxes1,
            bboxes2=bboxes2,
            mode='raw',
        ))

    sims = np.array(sims)
    if mode == 'max':
        return float(sims.max())
    elif mode == 'mean':
        return float(sims.mean())
    elif mode == 'raw':
        return sims.tolist()
    else:
        raise ValueError(f'Unknown similarity mode for bboxes - {mode!r}.')
