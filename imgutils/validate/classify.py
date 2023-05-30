from functools import lru_cache
from typing import Tuple, Optional, Dict, Union, Mapping

import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

from imgutils.data import rgb_encode, ImageTyping, load_image
from imgutils.utils import open_onnx_model

__all__ = [
    'anime_classify',
    'is_3d', 'is_bangumi', 'is_comic', 'is_illustration',
]

_MODEL_METAS = [
    ('mobilenetv3_large_100', 0.533, 0.438, 0.440, 0.446),
    ('mobilevitv2_150', 0.315, 0.354, 0.595, 0.511),
]
_LABELS = ['3d', 'bangumi', 'comic', 'illustration']

_MODEL_NAMES = [name for name, *_ in _MODEL_METAS]
_DEFAULT_MODEL_NAME = _MODEL_NAMES[0]
_MODEL_THRESHOLDS = {name: dict(zip(_LABELS, thresholds)) for name, *thresholds in _MODEL_METAS}


@lru_cache()
def _open_anime_classify_model(model_name):
    return open_onnx_model(hf_hub_download(
        f'deepghs/imgutils-models',
        f'anime_cls/anime_cls_{model_name}.onnx',
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


def _default_thresholds(model_name: str = _DEFAULT_MODEL_NAME) -> Mapping[str, float]:
    return _MODEL_THRESHOLDS[model_name]


def anime_classify(image: ImageTyping, model_name: str = _DEFAULT_MODEL_NAME,
                   check: bool = False, thresholds: Optional[Mapping[str, float]] = None) \
        -> Dict[str, Union[float, bool]]:
    image = load_image(image, force_background='white', mode='RGB')
    input_ = _img_encode(image)[None, ...]
    output, = _open_anime_classify_model(model_name).run(['output'], {'input': input_})
    values = dict(zip(_LABELS, map(lambda x: x.item(), output[0])))
    thresholds = thresholds or _default_thresholds(model_name)
    if check:
        return {label: values[label] >= thresholds[label] for label in _LABELS}
    else:
        return values


def _is_cls(image: ImageTyping, cls_name: str, model_name: str = _DEFAULT_MODEL_NAME, threshold: float = None):
    thresholds = dict(_default_thresholds(model_name))
    if thresholds is not None:
        thresholds[cls_name] = threshold

    return anime_classify(image, model_name, check=True, thresholds=thresholds)[cls_name]


def is_3d(image: ImageTyping, model_name: str = _DEFAULT_MODEL_NAME, threshold: float = None):
    return _is_cls(image, '3d', model_name, threshold)


def is_bangumi(image: ImageTyping, model_name: str = _DEFAULT_MODEL_NAME, threshold: float = None):
    return _is_cls(image, 'bangumi', model_name, threshold)


def is_comic(image: ImageTyping, model_name: str = _DEFAULT_MODEL_NAME, threshold: float = None):
    return _is_cls(image, 'comic', model_name, threshold)


def is_illustration(image: ImageTyping, model_name: str = _DEFAULT_MODEL_NAME, threshold: float = None):
    return _is_cls(image, 'illustration', model_name, threshold)
