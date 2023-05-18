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
from functools import lru_cache
from typing import List, Tuple

from huggingface_hub import hf_hub_download

from ._yolo import _image_preprocess, _data_postprocess
from ..data import ImageTyping, load_image, rgb_encode
from ..utils import open_onnx_model


@lru_cache()
def _open_head_detect_model(level: str = 's'):
    return open_onnx_model(hf_hub_download(
        'deepghs/imgutils-models',
        f'head_detect/head_detect_best_{level}.onnx'
    ))


def detect_heads(image: ImageTyping, level: str = 's', max_infer_size=640,
                 conf_threshold: float = 0.3, iou_threshold: float = 0.7) \
        -> List[Tuple[Tuple[int, int, int, int], str, float]]:
    """
    Overview:
        Detect human heads in anime images.

    :param image: Image to detect.
    :param level: The model level being used can be either `s` or `n`.
        The `n` model runs faster with smaller system overhead, while the `s` model achieves higher accuracy.
        The default value is `s`.
    :param max_infer_size: The maximum image size used for model inference, if the image size exceeds this limit,
        the image will be resized and used for inference. The default value is `640` pixels.
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
    image = load_image(image, mode='RGB')
    new_image, old_size, new_size = _image_preprocess(image, max_infer_size)

    data = rgb_encode(new_image)[None, ...]
    output, = _open_head_detect_model(level).run(['output0'], {'images': data})
    return _data_postprocess(output[0], conf_threshold, iou_threshold, old_size, new_size, ['head'])
