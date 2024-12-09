"""
Overview:
    Detect eyes in anime images.

    Trained on dataset `deepghs/anime_eye_detection <https://huggingface.co/datasets/deepghs/anime_eye_detection>`_ with YOLOv8.

    .. image:: eye_detect_demo.plot.py.svg
        :align: center

    This is an overall benchmark of all the eye detect models:

    .. image:: eye_detect_benchmark.plot.py.svg
        :align: center

"""
from typing import List, Tuple, Optional

from ..data import ImageTyping
from ..generic import yolo_predict

_REPO_ID = 'deepghs/anime_eye_detection'


def detect_eyes(image: ImageTyping, level: str = 's', version: str = 'v1.0', model_name: Optional[str] = None,
                conf_threshold: float = 0.3, iou_threshold: float = 0.3, **kwargs) \
        -> List[Tuple[Tuple[int, int, int, int], str, float]]:
    """
    Detect human eyes in anime images.

    This function uses a YOLOv8 model to detect eyes in the given anime image. It supports
    different model levels and versions, allowing for a trade-off between speed and accuracy.

    :param image: The input image for eye detection. Can be various image types supported by ImageTyping.
    :type image: ImageTyping

    :param level: The model level to use. Can be either 's' (for higher accuracy) or 'n' (for faster processing).
                  Default is 's'.
    :type level: str

    :param version: Version of the model to use. Default is 'v1.0'.
    :type version: str

    :param model_name: Optional custom model name. If not provided, it's constructed using version and level.
    :type model_name: Optional[str]

    :param conf_threshold: Confidence threshold for detections. Only detections with confidence above this
                           threshold are returned. Default is 0.3.
    :type conf_threshold: float

    :param iou_threshold: Intersection over Union (IoU) threshold for non-maximum suppression.
                          Detections with IoU above this threshold are considered overlapping and merged.
                          Default is 0.3.
    :type iou_threshold: float

    :return: A list of detected eyes. Each detection is represented by a tuple containing:
             - Bounding box coordinates as (x0, y0, x1, y1)
             - Detection class (always 'eye' for this function)
             - Confidence score of the detection
    :rtype: List[Tuple[Tuple[int, int, int, int], str, float]]

    :raises: May raise exceptions related to image loading or model prediction (from yolo_predict function).

    Examples::
        >>> from imgutils.detect import detect_eyes, detection_visualize
        >>>
        >>> image = 'squat.jpg'
        >>> result = detect_eyes(image)  # detect it
        >>> result
        [((297, 239, 341, 271), 'eye', 0.7760562896728516), ((230, 289, 263, 308), 'eye', 0.7682342529296875)]
        >>>
        >>> # visualize it
        >>> from matplotlib import pyplot as plt
        >>> plt.imshow(detection_visualize(image, result))
        >>> plt.show()
    """
    return yolo_predict(
        image=image,
        repo_id=_REPO_ID,
        model_name=model_name or f'eye_detect_{version}_{level}',
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        **kwargs,
    )
