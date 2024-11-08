"""
Overview:
    A tool for measuring the aesthetic level of anime images, with the model
    obtained from `skytnt/anime-aesthetic <https://huggingface.co/skytnt/anime-aesthetic>`_.

    .. image:: aesthetic_full.plot.py.svg
        :align: center

    This is an overall benchmark of all the operations in aesthetic models:

    .. image:: aesthetic_benchmark.plot.py.svg
        :align: center

    .. warning::
        These model is deprecated due to the poor effectiveness.
        Please use `imgutils.metrics.aesthetic.anime_dbaesthetic` for better evaluation.
"""

import cv2
import numpy as np
from PIL import Image
from deprecation import deprecated
from huggingface_hub import hf_hub_download

from ..config.meta import __VERSION__
from ..data import ImageTyping, load_image
from ..utils import open_onnx_model, ts_lru_cache

__all__ = [
    'get_aesthetic_score',
]


@ts_lru_cache()
def _open_aesthetic_model():
    return open_onnx_model(hf_hub_download(
        repo_id="skytnt/anime-aesthetic",
        filename="model.onnx"
    ))


def _preprocess(image: Image.Image):
    assert image.mode == 'RGB'
    img = np.array(image).astype(np.float32) / 255
    s = 768
    h, w = img.shape[:-1]
    h, w = (s, int(s * w / h)) if h > w else (int(s * h / w), s)
    ph, pw = s - h, s - w
    img_input = np.zeros([s, s, 3], dtype=np.float32)
    img_input[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w] = cv2.resize(img, (w, h))
    img_input = np.transpose(img_input, (2, 0, 1))
    return img_input[np.newaxis, :]


@deprecated(deprecated_in='0.4.2', removed_in='1.0.0', current_version=__VERSION__,
            details='Deprecated due to the low effectiveness.')
def get_aesthetic_score(image: ImageTyping):
    """
    Overview:
        Get aesthetic score for image.

    :param image: Original image.
    :return: Score of aesthetic.

    Examples::
        >>> from imgutils.metrics import get_aesthetic_score
        >>>
        >>> get_aesthetic_score('2053756.jpg')
        0.09986039996147156
        >>> get_aesthetic_score('1663584.jpg')
        0.24299287796020508
        >>> get_aesthetic_score('4886411.jpg')
        0.38091593980789185
        >>> get_aesthetic_score('2066024.jpg')
        0.5131649971008301
        >>> get_aesthetic_score('3670169.jpg')
        0.6011670827865601
        >>> get_aesthetic_score('5930006.jpg')
        0.7067991495132446
        >>> get_aesthetic_score('3821265.jpg')
        0.8237218260765076
        >>> get_aesthetic_score('5512471.jpg')
        0.9187621474266052
    """
    image = load_image(image, mode='RGB')
    retval, *_ = _open_aesthetic_model().run(None, {'img': _preprocess(image)})
    return float(retval.item())
