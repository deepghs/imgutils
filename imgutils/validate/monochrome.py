from functools import lru_cache
from typing import Optional, Tuple, Mapping

import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

from ..data import ImageTyping, load_image, rgb_encode
from ..utils import open_onnx_model

__all__ = [
    'get_monochrome_score',
    'is_monochrome',
]

_MODELS: Mapping[int, str] = {
    0: 'monochrome-caformer-110.onnx',
    2: 'monochrome-caformer_safe2-80.onnx',
    4: 'monochrome-caformer_safe4-70.onnx',
}


@lru_cache()
def _monochrome_validate_model(ckpt):
    return open_onnx_model(hf_hub_download(
        'deepghs/imgutils-models',
        f'monochrome/{ckpt}'
    ))


def _2d_encode(image: Image.Image, size: Tuple[int, int] = (384, 384),
               normalize: Optional[Tuple[float, float]] = (0.5, 0.5)):
    image = image.resize(size, Image.BILINEAR)
    data = rgb_encode(image, order_='CHW')

    if normalize is not None:
        mean_, std_ = normalize
        mean = np.asarray([mean_]).reshape((-1, 1, 1))
        std = np.asarray([std_]).reshape((-1, 1, 1))
        data = (data - mean) / std

    return data


def get_monochrome_score(image: ImageTyping, safe: int = 2) -> float:
    if safe not in _MODELS:
        raise ValueError(f'Safe level should be one of {set(sorted(_MODELS.keys()))!r}, but {safe!r} found.')

    image = load_image(image, mode='RGB')
    input_data = _2d_encode(image).astype(np.float32)
    input_data = np.stack([input_data])
    output_data, = _monochrome_validate_model(_MODELS[safe]).run(['output'], {'input': input_data})
    return float(output_data[0][1])


def is_monochrome(image: ImageTyping, threshold: float = 0.5, safe: int = 2) -> bool:
    return get_monochrome_score(image, safe) >= threshold
