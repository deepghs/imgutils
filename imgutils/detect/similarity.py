from typing import List, Literal, Union

import numpy as np

from .base import BBoxTyping, BBoxWithScoreAndLabel


def calculate_iou(box1: BBoxTyping, box2: BBoxTyping):
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
    print(iou_matrix)
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
