"""
Overview:
    Detect upper-half body in anime images.

    Trained on dataset `deepghs/anime_halfbody_detection <https://huggingface.co/datasets/deepghs/anime_halfbody_detection>`_ with YOLOv8.

    .. image:: halfbody_detect_demo.plot.py.svg
        :align: center

    This is an overall benchmark of all the halfbody detect models:

    .. image:: halfbody_detect_benchmark.plot.py.svg
        :align: center

    .. note::
        Please note that the primary purpose of this tool is to crop upper-body images from illustrations.
        Therefore, the training data used mostly consists of single-person images, and **the performance
        on images with multiple people is not guaranteed**. If you indeed need to process
        images with multiple people, the recommended approach is to first use
        the :func:`imgutils.detect.person.detect_person` function to crop individuals,
        and then use this tool to obtain upper-body images.

"""
from typing import List, Tuple, Optional

from ..data import ImageTyping
from ..generic import yolo_predict

_REPO_ID = 'deepghs/anime_halfbody_detection'


def detect_halfbody(image: ImageTyping, level: str = 's', version: str = 'v1.0', model_name: Optional[str] = None,
                    conf_threshold: float = 0.5, iou_threshold: float = 0.7, **kwargs) \
        -> List[Tuple[Tuple[int, int, int, int], str, float]]:
    """
    Detect human upper-half body in anime images.

    This function uses a YOLOv8 model to detect and localize upper-half bodies in anime-style images.
    It supports different model levels and versions for flexibility in speed and accuracy trade-offs.

    :param image: The input image to perform detection on. Can be a file path, URL, or image data.
    :type image: ImageTyping

    :param level: The model level to use. Can be either 's' (standard) or 'n' (nano).
                  The 'n' model is faster with lower system overhead, while 's' offers higher accuracy.
                  Default is 's'.
    :type level: str

    :param version: Version of the model to use. Default is 'v1.0'.
    :type version: str

    :param model_name: Optional custom model name. If not provided, it's constructed from version and level.
    :type model_name: Optional[str]

    :param conf_threshold: Confidence threshold for detections. Only detections with confidence above
                           this threshold are returned. Default is 0.5.
    :type conf_threshold: float

    :param iou_threshold: Intersection over Union (IoU) threshold for non-maximum suppression.
                          Overlapping detections above this threshold are merged. Default is 0.7.
    :type iou_threshold: float

    :return: A list of detections. Each detection is a tuple containing:
             - Bounding box coordinates as (x0, y0, x1, y1)
             - The string 'halfbody' (always)
             - The confidence score of the detection
    :rtype: List[Tuple[Tuple[int, int, int, int], str, float]]

    :raises ValueError: If an invalid level or version is provided.

    Examples::
        >>> from imgutils.detect import detect_halfbody, detection_visualize
        >>>
        >>> image = 'squat.jpg'
        >>> result = detect_halfbody(image)  # detect it
        >>> result
        [((127, 21, 629, 637), 'halfbody', 0.9040350914001465)]
        >>>
        >>> # visualize it
        >>> from matplotlib import pyplot as plt
        >>> plt.imshow(detection_visualize(image, result))
        >>> plt.show()
    """
    return yolo_predict(
        image=image,
        repo_id=_REPO_ID,
        model_name=model_name or f'halfbody_detect_{version}_{level}',
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        **kwargs,
    )
