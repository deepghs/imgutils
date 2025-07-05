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
- calculate_mask_iou: Computes IoU between two masks
- masks_similarity: Calculates similarities between two lists of masks
- detection_with_mask_similarity: Compares two lists of detections (with masks), considering both bounding masks and labels

This module is particularly useful for tasks involving object detection, 
image segmentation, and evaluation of detection algorithms.
"""

from typing import List, Literal, Union

import numpy as np

from .base import BBoxTyping, BBoxWithScoreAndLabel, MaskWithScoreAndLabel


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
    :raises ValueError: If an unknown mode is specified.

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
    m = len(bboxes1)
    n = len(bboxes2)

    if m == 0 and n == 0:
        if mode == 'raw':
            return []
        else:
            return 1.0

    iou_matrix = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            iou_matrix[i, j] = calculate_iou(bboxes1[i], bboxes2[j])

    # import here for faster launching speed
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    similarities = iou_matrix[row_ind, col_ind]

    max_len = max(m, n)
    padded_similarities = np.zeros(max_len)
    padded_similarities[:len(similarities)] = similarities

    if mode == 'max':
        return float(np.max(padded_similarities))
    elif mode == 'mean':
        return float(np.mean(padded_similarities))
    elif mode == 'raw':
        return padded_similarities.tolist()
    else:
        raise ValueError(f'Unknown similarity mode for bboxes - {mode!r}.')


def detection_similarity(detect1: List[Union[BBoxWithScoreAndLabel, MaskWithScoreAndLabel]],
                         detect2: List[Union[BBoxWithScoreAndLabel, MaskWithScoreAndLabel]],
                         mode: Literal['max', 'mean', 'raw'] = 'mean') -> Union[float, List[float]]:
    """
    Calculate the similarity between two lists of detections, considering both bounding boxes and labels.

    :param detect1: First list of detections, each containing a bounding box, label, and score.
    :type detect1: List[Union[BBoxWithScoreAndLabel, MaskWithScoreAndLabel]]
    :param detect2: Second list of detections, each containing a bounding box, label, and score.
    :type detect2: List[Union[BBoxWithScoreAndLabel, MaskWithScoreAndLabel]]
    :param mode: The mode for calculating similarity. Options are 'max', 'mean', or 'raw'. Defaults to 'mean'.
    :type mode: Literal['max', 'mean', 'raw']
    :return: The similarity score or list of scores, depending on the mode.
    :rtype: Union[float, List[float]]
    :raises ValueError: If an unknown mode is specified.

    This function compares two lists of detections by:

    1. Grouping detections by their labels.
    2. For each label, calculating the similarity between the corresponding bounding boxes.
    3. Aggregating the similarities based on the specified mode.

    The function processes detections label by label and combines their similarities.
    It's particularly useful for evaluating object detection results against ground truth.

    Example::
        >>> detect1 = [((0, 0, 2, 2), 'car', 0.9), ((3, 3, 5, 5), 'person', 0.8)]
        >>> detect2 = [((1, 1, 3, 3), 'car', 0.85), ((4, 4, 6, 6), 'person', 0.75)]
        >>> similarity = detection_similarity(detect1, detect2, mode='mean')
        >>> print(f"Mean detection similarity: {similarity:.4f}")
        Mean detection similarity: 0.1429
    """
    labels = sorted({*(l for _, l, *_ in detect1), *(l for _, l, *_ in detect2)})
    sims = []
    for current_label in labels:
        bboxes1 = [bbox for bbox, label, *_ in detect1 if label == current_label]
        bboxes2 = [bbox for bbox, label, *_ in detect2 if label == current_label]
        sims.extend(bboxes_similarity(bboxes1=bboxes1, bboxes2=bboxes2, mode='raw'))

    sims = np.array(sims)
    if mode == 'max':
        return float(sims.max())
    elif mode == 'mean':
        return float(sims.mean()) if sims.shape[0] > 0 else 1.0
    elif mode == 'raw':
        return sims.tolist()
    else:
        raise ValueError(f'Unknown similarity mode for bboxes - {mode!r}.')


def _mask_to_bool_mask(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Convert a mask array to a boolean mask.

    :param mask: The input mask array.
    :type mask: np.ndarray
    :param threshold: The threshold value for converting numeric masks to boolean. Defaults to 0.5.
    :type threshold: float
    :return: A boolean mask.
    :rtype: np.ndarray
    :raises TypeError: If the mask is not a numpy array or has an unsupported dtype.

    This function converts different types of mask arrays to boolean arrays:

    - If the mask is already a boolean array, it is returned as is.
    - If the mask is a numeric array, values >= threshold are set to True, others to False.

    Example::
        >>> import numpy as np
        >>> mask = np.array([[0.2, 0.7], [0.8, 0.3]])
        >>> bool_mask = _mask_to_bool_mask(mask)
        >>> print(bool_mask)
        [[False  True]
         [ True False]]
    """
    if isinstance(mask, np.ndarray):
        if np.issubdtype(mask.dtype, np.bool_):
            return mask
        elif np.issubdtype(mask.dtype, np.number):
            return mask >= threshold
        else:
            raise TypeError(f'Unknown dtype of the given mask - {mask.dtype!r}.')
    else:
        raise TypeError(f'Unknown type of the given mask - {mask!r}.')


def calculate_mask_iou(mask1: np.ndarray, mask2: np.ndarray, threshold: float = 0.5) -> float:
    """
    Calculate the Intersection over Union (IoU) between two masks.

    :param mask1: The first mask.
    :type mask1: np.ndarray
    :param mask2: The second mask.
    :type mask2: np.ndarray
    :param threshold: The threshold value for converting masks to boolean. Defaults to 0.5.
    :type threshold: float
    :return: The IoU value between the two masks.
    :rtype: float

    This function computes the IoU between two masks, which is defined as the area of intersection
    divided by the area of union. The masks are first converted to boolean arrays using the specified threshold.

    Example::
        >>> import numpy as np
        >>> mask1 = np.array([[1, 1], [1, 0]])
        >>> mask2 = np.array([[0, 1], [1, 1]])
        >>> iou = calculate_mask_iou(mask1, mask2)
        >>> print(f"IoU: {iou:.4f}")
        IoU: 0.5000
    """
    mask1 = _mask_to_bool_mask(mask1, threshold=threshold)
    mask2 = _mask_to_bool_mask(mask2, threshold=threshold)
    iou_value = ((mask1 & mask2).sum() / ((mask1 | mask2).sum() + 1e-6)).item()
    return iou_value


def masks_similarity(masks1: List[np.ndarray], masks2: List[np.ndarray],
                     mode: Literal['max', 'mean', 'raw'] = 'mean') -> Union[float, List[float]]:
    """
    Calculate the similarity between two lists of masks.

    :param masks1: First list of masks.
    :type masks1: List[np.ndarray]
    :param masks2: Second list of masks.
    :type masks2: List[np.ndarray]
    :param mode: The mode for calculating similarity. Options are 'max', 'mean', or 'raw'. Defaults to 'mean'.
    :type mode: Literal['max', 'mean', 'raw']
    :return: The similarity score or list of scores, depending on the mode.
    :rtype: Union[float, List[float]]
    :raises ValueError: If an unknown mode is specified.

    This function computes the similarity between two lists of masks using the Hungarian algorithm
    to find the optimal assignment. It then returns the similarity based on the specified mode:

    - ``max``: Returns the maximum IoU among all matched pairs.
    - ``mean``: Returns the average IoU of all matched pairs.
    - ``raw``: Returns a list of IoU values for all matched pairs.

    If both lists are empty, the function returns 1.0 for 'max' and 'mean' modes, and an empty list for 'raw' mode.

    Example::
        >>> import numpy as np
        >>> masks1 = [np.array([[1, 1], [1, 0]]), np.array([[0, 0], [1, 1]])]
        >>> masks2 = [np.array([[0, 1], [1, 1]]), np.array([[1, 0], [0, 0]])]
        >>> similarity = masks_similarity(masks1, masks2, mode='mean')
        >>> print(f"Mean similarity: {similarity:.4f}")
        Mean similarity: 0.5000
    """
    m, n = len(masks1), len(masks2)

    if m == 0 and n == 0:
        if mode == 'raw':
            return []
        else:
            return 1.0

    iou_matrix = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            iou_matrix[i, j] = calculate_mask_iou(masks1[i], masks2[j])

    # import here for faster launching speed
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    similarities = iou_matrix[row_ind, col_ind]

    max_len = max(m, n)
    padded_similarities = np.zeros(max_len)
    padded_similarities[:len(similarities)] = similarities

    if mode == 'max':
        return float(np.max(padded_similarities))
    elif mode == 'mean':
        return float(np.mean(padded_similarities))
    elif mode == 'raw':
        return padded_similarities.tolist()
    else:
        raise ValueError(f'Unknown similarity mode for masks - {mode!r}.')


def detection_with_mask_similarity(detect1: List[MaskWithScoreAndLabel], detect2: List[MaskWithScoreAndLabel],
                                   mode: Literal['max', 'mean', 'raw'] = 'mean') -> Union[float, List[float]]:
    """
    Calculate the similarity between two lists of mask detections, considering both masks and labels.

    :param detect1: First list of mask detections, each containing a label, score, and mask.
    :type detect1: List[MaskWithScoreAndLabel]
    :param detect2: Second list of mask detections, each containing a label, score, and mask.
    :type detect2: List[MaskWithScoreAndLabel]
    :param mode: The mode for calculating similarity. Options are 'max', 'mean', or 'raw'. Defaults to 'mean'.
    :type mode: Literal['max', 'mean', 'raw']
    :return: The similarity score or list of scores, depending on the mode.
    :rtype: Union[float, List[float]]
    :raises ValueError: If an unknown mode is specified.

    This function compares two lists of mask detections by:

    1. Grouping detections by their labels.
    2. For each label, calculating the similarity between the corresponding masks.
    3. Aggregating the similarities based on the specified mode.

    The function processes detections label by label and combines their similarities.
    It's particularly useful for evaluating instance segmentation results against ground truth.

    Example::
        >>> import numpy as np
        >>> # Example with simplified MaskWithScoreAndLabel format (_, label, score, mask)
        >>> detect1 = [(None, 'car', 0.9, np.array([[1, 1], [1, 0]])),
        ...            (None, 'person', 0.8, np.array([[0, 0], [1, 1]]))]
        >>> detect2 = [(None, 'car', 0.85, np.array([[0, 1], [1, 1]])),
        ...            (None, 'person', 0.75, np.array([[1, 0], [0, 0]]))]
        >>> similarity = detection_with_mask_similarity(detect1, detect2, mode='mean')
        >>> print(f"Mean detection similarity: {similarity:.4f}")
        Mean detection similarity: 0.2500
    """
    labels = sorted({*(l for _, l, *_ in detect1), *(l for _, l, *_ in detect2)})
    sims = []
    for current_label in labels:
        masks1 = [mask for _, label, _, mask in detect1 if label == current_label]
        masks2 = [mask for _, label, _, mask in detect2 if label == current_label]
        sims.extend(masks_similarity(masks1=masks1, masks2=masks2, mode='raw'))

    sims = np.array(sims)
    if mode == 'max':
        return float(sims.max())
    elif mode == 'mean':
        return float(sims.mean()) if sims.shape[0] > 0 else 1.0
    elif mode == 'raw':
        return sims.tolist()
    else:
        raise ValueError(f'Unknown similarity mode for bboxes - {mode!r}.')
