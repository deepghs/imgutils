"""
Overview:
    A model for classifying anime portrait images into 3 classes (``person``, ``halfbody``, ``head``).

    The following are sample images for testing.

    .. image:: portrait.plot.py.svg
        :align: center

    This is an overall benchmark of all the portrait classification models:

    .. image:: portrait_benchmark.plot.py.svg
        :align: center

    The models are hosted on
    `huggingface - deepghs/anime_portrait <https://huggingface.co/deepghs/anime_portrait>`_.
"""
import json
from functools import lru_cache
from typing import Tuple, Optional, Dict, List

import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

from imgutils.data import rgb_encode, ImageTyping, load_image
from imgutils.utils import open_onnx_model

__all__ = [
    'anime_portrait_score',
    'anime_portrait',
]

_DEFAULT_MODEL_NAME = 'mobilenetv3_v0_dist'


@lru_cache()
def _open_anime_portrait_model(model_name):
    return open_onnx_model(hf_hub_download(
        f'deepghs/anime_portrait',
        f'{model_name}/model.onnx',
    ))


@lru_cache()
def _get_anime_portrait_labels(model_name) -> List[str]:
    with open(hf_hub_download(
            f'deepghs/anime_portrait',
            f'{model_name}/meta.json',
    ), 'r') as f:
        return json.load(f)['labels']


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


def _raw_anime_portrait(image: ImageTyping, model_name: str = _DEFAULT_MODEL_NAME):
    image = load_image(image, force_background='white', mode='RGB')
    input_ = _img_encode(image)[None, ...]
    output, = _open_anime_portrait_model(model_name).run(['output'], {'input': input_})

    return output


def anime_portrait_score(image: ImageTyping, model_name: str = _DEFAULT_MODEL_NAME) -> Dict[str, float]:
    output = _raw_anime_portrait(image, model_name)
    values = dict(zip(_get_anime_portrait_labels(model_name), map(lambda x: x.item(), output[0])))
    return values


def anime_portrait(image: ImageTyping, model_name: str = _DEFAULT_MODEL_NAME) -> Tuple[str, float]:
    output = _raw_anime_portrait(image, model_name)[0]
    max_id = np.argmax(output)
    return _get_anime_portrait_labels(model_name)[max_id], output[max_id].item()
