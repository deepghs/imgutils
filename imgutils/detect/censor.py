"""
Overview:
    Detect human censor points (including female's nipples and genitals of both male and female) in anime images.

    Trained on dataset `deepghs/anime_censor_detection <https://huggingface.co/datasets/deepghs/anime_censor_detection>`_ with YOLOv8.

    .. collapse:: Overview of Censor Detect (NSFW Warning!!!)

        .. image:: censor_detect_demo.plot.py.svg
            :align: center

    This is an overall benchmark of all the censor detect models:

    .. image:: censor_detect_benchmark.plot.py.svg
        :align: center

"""
from typing import List, Tuple, Optional

from ..data import ImageTyping
from ..generic import yolo_predict

_REPO_ID = 'deepghs/anime_censor_detection'


def detect_censors(image: ImageTyping, level: str = 's', version: str = 'v1.0', model_name: Optional[str] = None,
                   conf_threshold: float = 0.3, iou_threshold: float = 0.7, **kwargs) \
        -> List[Tuple[Tuple[int, int, int, int], str, float]]:
    """
    Detect human censor points in anime images.

    This function uses pre-trained YOLOv8 models to identify and locate specific
    anatomical features that are typically censored in anime images. It can detect
    female nipples, male genitals, and female genitals.

    :param image: The input image to be analyzed. Can be a file path, URL, or image data.
    :type image: ImageTyping

    :param level: The model level to use, either 's' (standard) or 'n' (nano).
                  The 'n' model is faster but less accurate, while 's' is more accurate but slower.
    :type level: str

    :param version: The version of the model to use. Default is 'v1.0'.
    :type version: str

    :param model_name: Optional custom model name. If not provided, it will be constructed
                       from the version and level.
    :type model_name: Optional[str]

    :param conf_threshold: The confidence threshold for detections. Only detections with
                           confidence above this value will be returned. Default is 0.3.
    :type conf_threshold: float

    :param iou_threshold: The Intersection over Union (IoU) threshold for non-maximum
                          suppression. Detections with IoU above this value will be merged.
                          Default is 0.7.
    :type iou_threshold: float

    :return: A list of tuples, each containing:
             - A tuple of four integers (x0, y0, x1, y1) representing the bounding box
             - A string indicating the type of detection ('nipple_f', 'penis', or 'pussy')
             - A float representing the confidence score of the detection
    :rtype: List[Tuple[Tuple[int, int, int, int], str, float]]

    :raises ValueError: If an invalid level is provided.
    :raises RuntimeError: If the model fails to load or process the image.

    Examples::
        >>> from imgutils.detect import detect_censors, detection_visualize
        >>>
        >>> image = 'nude_girl.png'
        >>> result = detect_censors(image)  # detect it
        >>> result
        [
            ((365, 264, 399, 289), 'nipple_f', 0.7473511695861816),
            ((224, 260, 252, 285), 'nipple_f', 0.6830288171768188),
            ((206, 523, 240, 608), 'pussy', 0.6799028515815735)
        ]
        >>>
        >>> # visualize it
        >>> from matplotlib import pyplot as plt
        >>> plt.imshow(detection_visualize(image, result))
        >>> plt.show()
    """
    return yolo_predict(
        image=image,
        repo_id=_REPO_ID,
        model_name=model_name or f'censor_detect_{version}_{level}',
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        **kwargs,
    )
