"""
Overview:
    A tool for measuring the aesthetic level of anime images, with the model
    obtained from ` <https://huggingface.co/skytnt/anime-aesthetic>`_.

    This is an overall benchmark of all the operations in LPIPS models:

    .. image:: aesthetic.benchmark.py.svg
        :align: center
"""
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

from ..data import ImageTyping, load_image
from ..utils import open_onnx_model

__all__ = [
    'get_aesthetic_score',
]


@lru_cache()
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


def get_aesthetic_score(image: ImageTyping):
    image = load_image(image, mode='RGB')
    retval, *_ = _open_aesthetic_model().run(None, {'img': _preprocess(image)})
    return float(retval.item())
