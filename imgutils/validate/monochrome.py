from functools import lru_cache
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter
from PIL.Image import Resampling
from huggingface_hub import hf_hub_download
from scipy import signal

from ..data import ImageTyping, load_image, rgb_encode
from ..utils import open_onnx_model

__all__ = [
    'get_monochrome_score',
    'is_monochrome',
]

# _DEFAULT_MONOCHROME_CKPT = 'monochrome-resnet18-safe2-450.onnx'
_DEFAULT_MONOCHROME_CKPT = 'monochrome-levit_d0.2-500.onnx'


@lru_cache()
def _monochrome_validate_model(ckpt):
    return open_onnx_model(hf_hub_download(
        'deepghs/imgutils-models',
        f'monochrome/{ckpt}'
    ))


def np_hist(x, a_min: float = 0.0, a_max: float = 1.0, bins: int = 256):
    x = np.asarray(x)
    edges = np.linspace(a_min, a_max, bins + 1)
    cnt, _ = np.histogram(x, bins=edges)
    return cnt / cnt.sum()


def butterworth_filter(r, fc):
    w = fc / (len(r) / 2)  # Normalize the frequency
    b, a = signal.butter(5, w, 'low')
    return np.clip(signal.filtfilt(b, a, r), a_min=0.0, a_max=1.0)


def _hsv_encode(image: Image.Image, feature_bins: int = 180, mf: Optional[int] = 5,
                maxpixels: int = 20000, fc: Optional[int] = 75, normalize: bool = True):
    if image.width * image.height > maxpixels:
        r = (image.width * image.height / maxpixels) ** 0.5
        new_width, new_height = map(lambda x: int(round(x / r)), image.size)
        image = image.resize((new_width, new_height))

    if mf is not None:
        image = image.filter(ImageFilter.MedianFilter(mf))
    image = image.convert('HSV')

    data = (np.transpose(np.asarray(image), (2, 0, 1)) / 255.0).astype(np.float32)
    channels = [np_hist(data[i], bins=feature_bins) for i in range(3)]
    if fc is not None:
        channels = [butterworth_filter(ch, fc) for ch in channels]

    dist = np.stack(channels)
    assert dist.shape == (3, feature_bins)

    if normalize:
        mean = np.mean(dist, axis=1, keepdims=True)
        std = np.std(dist, axis=1, keepdims=True, ddof=1)
        dist = (dist - mean) / std

    return dist


def _2d_encode(image: Image.Image, size: Tuple[int, int] = (384, 384),
               normalize: Optional[Tuple[float, float]] = (0.5, 0.5)):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(size, Resampling.BILINEAR)
    data = rgb_encode(image, order_='CHW')

    if normalize is not None:
        mean_, std_ = normalize
        mean = np.asarray([mean_]).reshape((-1, 1, 1))
        std = np.asarray([std_]).reshape((-1, 1, 1))
        data = (data - mean) / std

    return data


def get_monochrome_score(image: ImageTyping, ckpt: str = _DEFAULT_MONOCHROME_CKPT) -> float:
    image = load_image(image, mode='RGB')
    # input_data = _hsv_encode(image).astype(np.float32)
    input_data = _2d_encode(image).astype(np.float32)
    input_data = np.stack([input_data])
    output_data, = _monochrome_validate_model(ckpt).run(['output'], {'input': input_data})
    return float(output_data[0][1])


def is_monochrome(image: ImageTyping, threshold: float = 0.5, ckpt: str = _DEFAULT_MONOCHROME_CKPT) -> bool:
    return get_monochrome_score(image, ckpt) >= threshold
