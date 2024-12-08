"""
Overview:
    This module provides functionality for detecting human bodies (including the entire body) in anime images.
    It uses YOLOv8 models trained on the `AniDet3 <https://universe.roboflow.com/university-of-michigan-ann-arbor/anidet3-ai42v>`_
    dataset from Roboflow.

    .. image:: person_detect_demo.plot.py.svg
        :align: center

    The module includes a main function :func:`detect_person` for performing the detection task,
    and utilizes the `yolo_predict` function from the generic module for the actual prediction.

    The module supports different model levels and versions, allowing users to choose
    between speed and accuracy based on their requirements.

    This is an overall benchmark of all the person detect models:

    .. image:: person_detect_benchmark.plot.py.svg
        :align: center

"""
from typing import Optional

from ..data import ImageTyping
from ..generic import yolo_predict

_REPO_ID = 'deepghs/anime_person_detection'


def detect_person(image: ImageTyping, level: str = 'm', version: str = 'v1.1', model_name: Optional[str] = None,
                  conf_threshold: float = 0.3, iou_threshold: float = 0.5, **kwargs):
    """
    Detect human bodies (including the entire body) in anime images.

    This function uses YOLOv8 models to detect human bodies in anime-style images.
    It supports different model levels and versions, allowing users to balance between
    detection speed and accuracy.

    :param image: The input image for detection. Can be various types as defined by ImageTyping.
    :type image: ImageTyping

    :param level: The model level to use. Options are 'n', 's', 'm', or 'x'.
                  'n' is fastest but less accurate, 'x' is most accurate but slower.
    :type level: str

    :param version: The version of the model to use. Available versions are 'v0', 'v1', and 'v1.1'.
    :type version: str

    :param model_name: Optional custom model name. If provided, overrides the auto-generated model name.
    :type model_name: Optional[str]

    :param conf_threshold: Confidence threshold for detections. Only detections with
                           confidence above this value are returned.
    :type conf_threshold: float

    :param iou_threshold: Intersection over Union (IoU) threshold for non-maximum suppression.
    :type iou_threshold: float

    :return: A list of detection results. Each result is a tuple containing:
             ((x0, y0, x1, y1), 'person', confidence_score)
    :rtype: List[Tuple[Tuple[int, int, int, int], str, float]]

    :raises ValueError: If an invalid level or version is provided.

    Example:
        >>> from imgutils.detect import detect_person, detection_visualize
        >>> image = 'genshin_post.jpg'
        >>> result = detect_person(image)
        >>> print(result)
        [
            ((371, 232, 564, 690), 'person', 0.7533698678016663),
            ((30, 135, 451, 716), 'person', 0.6788613796234131),
            ((614, 393, 830, 686), 'person', 0.5612757205963135),
            ((614, 3, 1275, 654), 'person', 0.4047100841999054)
        ]

    .. note::
        For visualization of results, you can use the :func:`imgutils.detect.visual.detection_visualize` function.
    """
    return yolo_predict(
        image=image,
        repo_id=_REPO_ID,
        model_name=model_name or f'person_detect_{version}_{level}',
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        **kwargs,
    )
