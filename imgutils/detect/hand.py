"""
Overview:
    This module provides functionality for detecting human hands in anime images.

    .. image:: hand_detect_demo.plot.py.svg
        :align: center

    It uses YOLOv8 models trained on the
    `deepghs/anime_hand_detection <https://huggingface.co/datasets/deepghs/anime_hand_detection>`_ dataset
    from HuggingFace. The module offers a main function :func:`detect_hands` for hand detection in anime images,
    with options to choose different model levels and versions for balancing speed and accuracy.

    This is an overall benchmark of all the hand detect models:

    .. image:: hand_detect_benchmark.plot.py.svg
        :align: center

"""
from typing import List, Tuple, Optional

from ..data import ImageTyping
from ..generic import yolo_predict

_REPO_ID = 'deepghs/anime_hand_detection'


def detect_hands(image: ImageTyping, level: str = 's', version: str = 'v1.0', model_name: Optional[str] = None,
                 conf_threshold: float = 0.35, iou_threshold: float = 0.7, **kwargs) \
        -> List[Tuple[Tuple[int, int, int, int], str, float]]:
    """
    Detect human hand points in anime images.

    This function uses a YOLOv8 model to detect hands in the given anime image. It allows for
    configuration of the model level, version, and detection thresholds to suit different use cases.

    :param image: Image to detect. Can be various types as defined by ImageTyping.
    :type image: ImageTyping

    :param level: The model level being used, either 's' or 'n'.
                  's' (standard) offers higher accuracy, while 'n' (nano) provides faster processing.
    :type level: str

    :param version: Version of the model to use. Default is 'v1.0'.
    :type version: str

    :param model_name: Optional custom model name. If not provided, it's constructed from version and level.
    :type model_name: Optional[str]

    :param conf_threshold: Confidence threshold for detections. Only detections with confidence
                           above this value are returned. Default is 0.35.
    :type conf_threshold: float

    :param iou_threshold: Intersection over Union (IOU) threshold for non-maximum suppression.
                          Detections with IOU above this value are considered overlapping and merged.
                          Default is 0.7.
    :type iou_threshold: float

    :return: A list of detection results. Each result is a tuple containing:
             - Bounding box coordinates as (x0, y0, x1, y1)
             - Class label (always 'hand' for this function)
             - Confidence score
    :rtype: List[Tuple[Tuple[int, int, int, int], str, float]]

    :raises: May raise exceptions related to image loading or model inference.

    Example:
        >>> from PIL import Image
        >>> image = Image.open('anime_image.jpg')
        >>> results = detect_hands(image, level='s', conf_threshold=0.4)
        >>> for bbox, label, conf in results:
        ...     print(f"Hand detected at {bbox} with confidence {conf}")
    """
    return yolo_predict(
        image=image,
        repo_id=_REPO_ID,
        model_name=model_name or f'hand_detect_{version}_{level}',
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        **kwargs,
    )
