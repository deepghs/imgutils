from functools import lru_cache
from typing import Optional, Tuple

import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

from ..data import ImageTyping, load_image, rgb_encode
from ..utils import open_onnx_model

__all__ = [
    'get_monochrome_score',
    'is_monochrome',
]

_DEFAULT_MONOCHROME_CKPT = 'monochrome-caformer_safe2-80.onnx'


@lru_cache()
def _monochrome_validate_model(ckpt):
    return open_onnx_model(hf_hub_download(
        'deepghs/imgutils-models',
        f'monochrome/{ckpt}'
    ))


def _2d_encode(image: Image.Image, size: Tuple[int, int] = (384, 384),
               normalize: Optional[Tuple[float, float]] = (0.5, 0.5)):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(size, Image.BILINEAR)
    data = rgb_encode(image, order_='CHW')

    if normalize is not None:
        mean_, std_ = normalize
        mean = np.asarray([mean_]).reshape((-1, 1, 1))
        std = np.asarray([std_]).reshape((-1, 1, 1))
        data = (data - mean) / std

    return data


def get_monochrome_score(image: ImageTyping, ckpt: str = _DEFAULT_MONOCHROME_CKPT) -> float:
    image = load_image(image, mode='RGB')
    input_data = _2d_encode(image).astype(np.float32)
    input_data = np.stack([input_data])
    output_data, = _monochrome_validate_model(ckpt).run(['output'], {'input': input_data})
    return float(output_data[0][1])


def is_monochrome(image: ImageTyping, threshold: float = 0.5, ckpt: str = _DEFAULT_MONOCHROME_CKPT) -> bool:
    return get_monochrome_score(image, ckpt) >= threshold
