"""
Overview:
    This module provides functionality for detecting human heads in anime images using YOLOv8 models.

    .. image:: head_detect_demo.plot.py.svg
        :align: center

    Key Features:

    - Head detection in anime images
    - Support for different model levels ('s' for accuracy, 'n' for speed)
    - Customizable confidence and IoU thresholds
    - Integration with Hugging Face model repository

    The module is based on the `deepghs/anime_head_detection <https://huggingface.co/datasets/deepghs/anime_head_detection>`_
    dataset contributed by our developers and uses YOLOv8/YOLO11 architecture for object detection.

    Example usage and benchmarks are provided in the module overview.

    This is an overall benchmark of all the head detect models:

    .. image:: head_detect_benchmark.plot.py.svg
        :align: center

"""
import warnings
from typing import List, Tuple, Optional

from ..data import ImageTyping
from ..generic import yolo_predict

_REPO_ID = 'deepghs/anime_head_detection'


def detect_heads(image: ImageTyping, level: Optional[str] = None,
                 model_name: Optional[str] = 'head_detect_v2.0_s',
                 conf_threshold: float = 0.4, iou_threshold: float = 0.7, **kwargs) \
        -> List[Tuple[Tuple[int, int, int, int], str, float]]:
    """
    Detect human heads in anime images using YOLOv8 models.

    This function applies a pre-trained YOLOv8 model to detect human heads in the given anime image.
    It supports different model levels and allows customization of detection parameters.

    :param image: The input image for head detection. Can be a file path, URL, or image data.
    :type image: ImageTyping

    :param level: The model level to use. 's' for higher accuracy, 'n' for faster speed.
                  Default is None (actually equals to 's').
                  This parameter is deprecated and will be removed in future versions.
    :type level: Optional[str]

    :param model_name: Name of the specific YOLO model to use. If not provided, uses 'head_detect_v2.0_s'.
    :type model_name: Optional[str]

    :param conf_threshold: Confidence threshold for detection results. Only detections with confidence above this value are returned. Default is 0.4.
    :type conf_threshold: float

    :param iou_threshold: IoU (Intersection over Union) threshold for non-maximum suppression. Helps in removing overlapping detections. Default is 0.7.
    :type iou_threshold: float

    :return: A list of detected heads. Each item is a tuple containing:
             - Bounding box coordinates as (x0, y0, x1, y1)
             - Class label (always 'head' for this function)
             - Confidence score of the detection
    :rtype: List[Tuple[Tuple[int, int, int, int], str, float]]

    :raises ValueError: If an invalid level is provided or if there's an issue with the model loading or prediction.

    Example:
        >>> from imgutils.detect import detect_heads, detection_visualize
        >>> image = 'mostima_post.jpg'
        >>> result = detect_heads(image)
        >>> print(result)
        [((29, 441, 204, 584), 'head', 0.7874319553375244),
         ((346, 59, 529, 275), 'head', 0.7510495185852051),
         ((606, 51, 895, 336), 'head', 0.6986488103866577)]

    .. note::

        For visualization of results, you can use the :func:`imgutils.detect.visual.detection_visualize` function.

    .. warning::

        The 'level' parameter is deprecated and will be removed in future versions. Use 'model_name' instead.
    """
    if level:
        warnings.warn(DeprecationWarning(
            'Argument level in function detect_heads is deprecated and will be removed in the future, '
            'please migrate to model_name as soon as possible.'
        ))

    return yolo_predict(
        image=image,
        repo_id=_REPO_ID,
        model_name=model_name or f'head_detect_v0_{level or "s"}',
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        **kwargs,
    )
