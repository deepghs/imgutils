"""
Overview:
    Detect human heads (including the entire head) in anime images.

    Trained on dataset `ani_face_detection <https://universe.roboflow.com/linog/ani_face_detection>`_ with YOLOv8.

    .. image:: head_detect_demo.plot.py.svg
        :align: center

    This is an overall benchmark of all the head detect models:

    .. image:: head_detect_benchmark.plot.py.svg
        :align: center

"""
from typing import List, Tuple, Optional

from ..data import ImageTyping
from ..generic import yolo_predict

_REPO_ID = 'deepghs/anime_head_detection'


def detect_heads(image: ImageTyping, level: str = 's', model_name: Optional[str] = None,
                 conf_threshold: float = 0.3, iou_threshold: float = 0.7) \
        -> List[Tuple[Tuple[int, int, int, int], str, float]]:
    """
    Overview:
        Detect human heads in anime images.

    :param image: Image to detect.
    :param level: The model level being used can be either `s` or `n`.
        The `n` model runs faster with smaller system overhead, while the `s` model achieves higher accuracy.
        The default value is `s`.
    :param model_name: Name of the YOLO model to use, use v0 models with optional `level` when not assigned.
    :type model_name: str, optional
    :param conf_threshold: The confidence threshold, only detection results with confidence scores above
        this threshold will be returned. The default value is `0.3`.
    :param iou_threshold: The detection area coverage overlap threshold, areas with overlaps above this threshold
        will be discarded. The default value is `0.7`.
    :return: The detection results list, each item includes the detected area `(x0, y0, x1, y1)`,
        the target type (always `head`) and the target confidence score.

    Examples::
        >>> from imgutils.detect import detect_heads, detection_visualize
        >>>
        >>> image = 'mostima_post.jpg'
        >>> result = detect_heads(image)  # detect it
        >>> result
        [
            ((29, 441, 204, 584), 'head', 0.7874319553375244),
            ((346, 59, 529, 275), 'head', 0.7510495185852051),
            ((606, 51, 895, 336), 'head', 0.6986488103866577)
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
        model_name=model_name or f'head_detect_v0_{level}',
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
    )
