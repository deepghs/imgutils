"""
Overview:
    Detect human hands in anime images.

    Trained on dataset `deepghs/anime_hand_detection <https://huggingface.co/datasets/deepghs/anime_hand_detection>`_ with YOLOv8.

    .. image:: hand_detect_demo.plot.py.svg
        :align: center

    This is an overall benchmark of all the hand detect models:

    .. image:: hand_detect_benchmark.plot.py.svg
        :align: center

"""
from typing import List, Tuple, Optional

from ..data import ImageTyping
from ..generic import yolo_predict

_REPO_ID = 'deepghs/anime_hand_detection'


def detect_hands(image: ImageTyping, level: str = 's', version: str = 'v1.0', model_name: Optional[str] = None,
                 conf_threshold: float = 0.35, iou_threshold: float = 0.7) \
        -> List[Tuple[Tuple[int, int, int, int], str, float]]:
    """
    Overview:
        Detect human hand points in anime images.

    :param image: Image to detect.
    :param level: The model level being used can be either `s` or `n`.
        The `n` model runs faster with smaller system overhead, while the `s` model achieves higher accuracy.
        The default value is `s`.
    :param version: Version of model, default is ``v1.0``.
    :param max_infer_size: The maximum image size used for model inference, if the image size exceeds this limit,
        the image will be resized and used for inference. The default value is `640` pixels.
    :param conf_threshold: The confidence threshold, only detection results with confidence scores above
        this threshold will be returned. The default value is `0.3`.
    :param iou_threshold: The detection area coverage overlap threshold, areas with overlaps above this threshold
        will be discarded. The default value is `0.7`.
    :return: The detection results list, each item includes the detected area `(x0, y0, x1, y1)`,
        the target type (always `hand`) and the target confidence score.
    """
    return yolo_predict(
        image=image,
        repo_id=_REPO_ID,
        model_name=model_name or f'hand_detect_{version}_{level}',
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
    )
