"""
Overview:
    A model for classifying anime images into 4 classes (``3d``, ``bangumi``, ``comic`` and ``illustration``).

    The following are sample images for testing.

    .. image:: classify.plot.py.svg
        :align: center

    This is an overall benchmark of all the classification validation models:

    .. image:: classify_benchmark.plot.py.svg
        :align: center

    The models are hosted on
    `huggingface - deepghs/anime_classification <https://huggingface.co/deepghs/anime_classification>`_.
"""
from functools import lru_cache
from typing import Tuple, Optional, Dict

import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

from imgutils.data import rgb_encode, ImageTyping, load_image
from imgutils.utils import open_onnx_model

__all__ = [
    'anime_classify_scores',
    'anime_classify',
]

_LABELS = ['3d', 'bangumi', 'comic', 'illustration']
_MODEL_NAMES = [
    'caformer_s36',
    'caformer_s36_plus',
    'mobilenetv3',
    'mobilenetv3_sce',
    'mobilevitv2_150',
]
_DEFAULT_MODEL_NAME = 'mobilenetv3_sce'


@lru_cache()
def _open_anime_classify_model(model_name):
    return open_onnx_model(hf_hub_download(
        f'deepghs/anime_classification',
        f'{model_name}/model.onnx',
    ))


def _img_encode(image: Image.Image, size: Tuple[int, int] = (384, 384),
                normalize: Optional[Tuple[float, float]] = (0.5, 0.5)):
    image = image.resize(size, Image.BILINEAR)
    data = rgb_encode(image, order_='CHW')

    if normalize is not None:
        mean_, std_ = normalize
        mean = np.asarray([mean_]).reshape((-1, 1, 1))
        std = np.asarray([std_]).reshape((-1, 1, 1))
        data = (data - mean) / std

    return data.astype(np.float32)


def _raw_anime_classify(image: ImageTyping, model_name: str = _DEFAULT_MODEL_NAME):
    image = load_image(image, force_background='white', mode='RGB')
    input_ = _img_encode(image)[None, ...]
    output, = _open_anime_classify_model(model_name).run(['output'], {'input': input_})

    return output


def anime_classify_scores(image: ImageTyping, model_name: str = _DEFAULT_MODEL_NAME) \
        -> Dict[str, float]:
    output = _raw_anime_classify(image, model_name)
    values = dict(zip(_LABELS, map(lambda x: x.item(), output[0])))
    return values


def anime_classify(image: ImageTyping, model_name: str = _DEFAULT_MODEL_NAME) -> Tuple[str, float]:
    output = _raw_anime_classify(image, model_name)[0]
    max_id = np.argmax(output)
    return _LABELS[max_id], output[max_id].item()
