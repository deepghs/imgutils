"""
Overview:
    Detect human faces in anime images.

    Trained on dataset `Anime Face CreateML <https://universe.roboflow.com/my-workspace-mph8o/anime-face-createml>`_
    with YOLOv8.

    .. image:: face_detect_demo.plot.py.svg
        :align: center

    This is an overall benchmark of all the face detect models:

    .. image:: face_detect_benchmark.plot.py.svg
        :align: center

    The models are hosted on
    `huggingface - deepghs/anime_face_detection <https://huggingface.co/deepghs/anime_face_detection>`_.

"""
from functools import lru_cache
from typing import List, Tuple

from huggingface_hub import hf_hub_download

from ._yolo import _image_preprocess, _data_postprocess
from ..data import ImageTyping, load_image, rgb_encode
from ..utils import open_onnx_model


@lru_cache()
def _open_face_detect_model(level: str = 's', version: str = 'v1.4'):
    return open_onnx_model(hf_hub_download(
        f'deepghs/anime_face_detection',
        f'face_detect_{version}_{level}/model.onnx'
    ))


def detect_faces(image: ImageTyping, level: str = 's', version: str = 'v1.4', max_infer_size=640,
                 conf_threshold: float = 0.25, iou_threshold: float = 0.7) \
        -> List[Tuple[Tuple[int, int, int, int], str, float]]:
    """
    Overview:
        Detect human faces in anime images.

    :param image: Image to detect.
    :param level: The model level being used can be either `s` or `n`.
        The `n` model runs faster with smaller system overface, while the `s` model achieves higher accuracy.
        The default value is `s`.
    :param version: Version of model, default is ``v1.4``.
        Available versions are ``v0``, ``v1``, ``v1.3`` and ``v1.4``.
    :param max_infer_size: The maximum image size used for model inference, if the image size exceeds this limit,
        the image will be resized and used for inference. The default value is `640` pixels.
    :param conf_threshold: The confidence threshold, only detection results with confidence scores above
        this threshold will be returned. The default value is `0.25`.
    :param iou_threshold: The detection area coverage overlap threshold, areas with overlaps above this threshold
        will be discarded. The default value is `0.7`.
    :return: The detection results list, each item includes the detected area `(x0, y0, x1, y1)`,
        the target type (always `face`) and the target confidence score.

    Examples::
        >>> from imgutils.detect import detect_faces, detection_visualize
        >>>
        >>> image = 'mostima_post.jpg'
        >>> result = detect_faces(image)  # detect it
        >>> result
        [
            ((29, 441, 204, 584), 'face', 0.7874319553375244),
            ((346, 59, 529, 275), 'face', 0.7510495185852051),
            ((606, 51, 895, 336), 'face', 0.6986488103866577)
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
    output, = _open_face_detect_model(level, version).run(['output0'], {'images': data})
    return _data_postprocess(output[0], conf_threshold, iou_threshold, old_size, new_size, ['face'])
