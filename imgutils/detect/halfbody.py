"""
Overview:
    Detect human halfbody points (including female's nipples and genitals of both male and female) in anime images.

    Trained on dataset `deepghs/anime_halfbody_detection <https://huggingface.co/datasets/deepghs/anime_halfbody_detection>`_ with YOLOv8.

    .. image:: halfbody_detect_demo.plot.py.svg
        :align: center

    This is an overall benchmark of all the halfbody detect models:

    .. image:: halfbody_detect_benchmark.plot.py.svg
        :align: center

"""
from functools import lru_cache
from typing import List, Tuple

from huggingface_hub import hf_hub_download

from ._yolo import _image_preprocess, _data_postprocess
from ..data import ImageTyping, load_image, rgb_encode
from ..utils import open_onnx_model


@lru_cache()
def _open_halfbody_detect_model(level: str = 's', version: str = 'v1.0'):
    return open_onnx_model(hf_hub_download(
        f'deepghs/anime_halfbody_detection',
        f'halfbody_detect_{version}_{level}/model.onnx'
    ))


_LABELS = ["halfbody"]


def detect_halfbodies(image: ImageTyping, level: str = 's', version: str = 'v0.4', max_infer_size=640,
                      conf_threshold: float = 0.5, iou_threshold: float = 0.7) \
        -> List[Tuple[Tuple[int, int, int, int], str, float]]:
    """
    Overview:
        Detect human halfbodies in anime images.

    :param image: Image to detect.
    :param level: The model level being used can be either `s` or `n`.
        The `n` model runs faster with smaller system overhead, while the `s` model achieves higher accuracy.
        The default value is `s`.
    :param version: Version of model, default is ``v1.0``.
    :param max_infer_size: The maximum image size used for model inference, if the image size exceeds this limit,
        the image will be resized and used for inference. The default value is `640` pixels.
    :param conf_threshold: The confidence threshold, only detection results with confidence scores above
        this threshold will be returned. The default value is `0.5`.
    :param iou_threshold: The detection area coverage overlap threshold, areas with overlaps above this threshold
        will be discarded. The default value is `0.7`.
    :return: The detection results list, each item includes the detected area `(x0, y0, x1, y1)`,
        the target type (always `halfbody`) and the target confidence score.

    Examples::
        >>> from imgutils.detect import detect_halfbodies, detection_visualize
        >>>
        >>> image = 'nude_girl.png'
        >>> result = detect_halfbodies(image)  # detect it
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
    image = load_image(image, mode='RGB')
    new_image, old_size, new_size = _image_preprocess(image, max_infer_size)

    data = rgb_encode(new_image)[None, ...]
    output, = _open_halfbody_detect_model(level, version).run(['output0'], {'images': data})
    return _data_postprocess(output[0], conf_threshold, iou_threshold, old_size, new_size, _LABELS)