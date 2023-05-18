"""
Overview:
    Detect human bodies (including the entire body) in anime images.

    Trained on dataset `AniDet3 <https://universe.roboflow.com/university-of-michigan-ann-arbor/anidet3-ai42v>`_ \
        with YOLOv8.

    .. image:: person_detect_demo.plot.py.svg
        :align: center

    This is an overall benchmark of all the person detect models:

    .. image:: person_detect_benchmark.plot.py.svg
        :align: center

"""
from functools import lru_cache

from huggingface_hub import hf_hub_download

from ._yolo import _image_preprocess, _data_postprocess
from ..data import ImageTyping, load_image, rgb_encode
from ..utils import open_onnx_model

_VERSIONS = {
    'v0': '',
    'v1': 'plus_',
    'v1.1': 'plus_v1.1_',
}


@lru_cache()
def _open_person_detect_model(level: str, version: str):
    return open_onnx_model(hf_hub_download(
        'deepghs/imgutils-models',
        f'person_detect/person_detect_{_VERSIONS[version]}best_{level}.onnx'
    ))


def detect_person(image: ImageTyping, level: str = 'm', version: str = 'v1.1', max_infer_size=640,
                  conf_threshold: float = 0.3, iou_threshold: float = 0.5):
    """
    Overview:
        Detect human bodies (including the entire body) in anime images.

    :param image: Image to detect.
    :param level: The model level being used can be either ``n``, ``s``, ``m`` or ``x``.
        The ``n`` model runs faster with smaller system overhead, while the ``m`` model achieves higher accuracy.
        The default value is ``m``.
    :param version: Version of model, default is ``v1.1``. Available versions are ``v0``, ``v1`` and ``v1.1``.
    :param max_infer_size: The maximum image size used for model inference, if the image size exceeds this limit,
        the image will be resized and used for inference. The default value is ``640`` pixels.
    :param conf_threshold: The confidence threshold, only detection results with confidence scores above
        this threshold will be returned. The default value is `0.3`.
    :param iou_threshold: The detection area coverage overlap threshold, areas with overlaps above this threshold
        will be discarded. The default value is `0.5`.
    :return: The detection results list, each item includes the detected area `(x0, y0, x1, y1)`,
        the target type (always `person`) and the target confidence score.

    Examples::
        >>> from imgutils.detect import detect_person, detection_visualize
        >>>
        >>> image = 'genshin_post.jpg'
        >>> result = detect_person(image)
        >>> result
        [
            ((371, 232, 564, 690), 'person', 0.7533698678016663),
            ((30, 135, 451, 716), 'person', 0.6788613796234131),
            ((614, 393, 830, 686), 'person', 0.5612757205963135),
            ((614, 3, 1275, 654), 'person', 0.4047100841999054)
        ]
        >>>
        >>> # visualize it
        >>> from matplotlib import pyplot as plt
        >>> plt.imshow(detection_visualize(image, result))
        >>> plt.show()

    .. note::
        Please note that certain combinations of versions and levels may not have corresponding models.
        When using them, please refer to the performance chart at the top of that page, which lists
        the versions and models included.

    """
    image = load_image(image, mode='RGB')
    new_image, old_size, new_size = _image_preprocess(image, max_infer_size)

    data = rgb_encode(new_image)[None, ...]
    output, = _open_person_detect_model(level, version).run(['output0'], {'images': data})
    return _data_postprocess(output[0], conf_threshold, iou_threshold, old_size, new_size, ['person'])
