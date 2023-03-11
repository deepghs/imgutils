from functools import lru_cache
from typing import Tuple, Sequence

import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN
from tqdm.auto import tqdm

from ..data import load_image, ImageTyping


def _image_resize(image: Image.Image, size=640):
    return image.resize((size, size), resample=Image.BILINEAR)


def _fft_encode(image: Image.Image, size=640, scale=(0.25, 0.75)):
    image = _image_resize(image, size)
    data = np.asarray(image)
    height, width, channels = data.shape
    ls, us = scale

    fshift = np.fft.fftshift(np.fft.fft2(data))
    fft_freq = np.log(np.abs(fshift) + 1e-6)
    cropped_freq = fft_freq[int(height * ls): int(height * us), int(width * ls): int(width * us)]
    blur_freq = cv2.GaussianBlur(cropped_freq, (5, 5), 0)
    return blur_freq


def _diff(d1: np.ndarray, d2: np.ndarray) -> float:
    return float(((d1 - d2) ** 2).mean())


def fft_difference(img1: ImageTyping, img2: ImageTyping, size: int = 640, scale: Tuple[float, float] = (0.25, 0.75)):
    image1 = load_image(img1, mode='RGB')
    image2 = load_image(img2, mode='RGB')
    return _diff(
        _fft_encode(image1, size=size, scale=scale),
        _fft_encode(image2, size=size, scale=scale),
    )


def fft_clustering(imgs: Sequence[ImageTyping], size: int = 640, scale: Tuple[float, float] = (0.15, 0.75),
                   threshold: float = 0.3, min_samples: int = 2):
    images = [load_image(item, mode='RGB') for item in imgs]
    n = len(images)
    encoded = [_fft_encode(item, size, scale) for item in tqdm(images)]
    stacked = np.stack(encoded)  # BHWC
    print(stacked)
    print(stacked.shape)

    @lru_cache(maxsize=min(n * (n - 1) // 2, 10000))
    def _get_diff(x, y):
        return _diff(stacked[x], stacked[y])

    def _metric(x, y):
        x, y = int(min(x, y)), int(max(x, y))
        return _get_diff(x, y)

    samples = np.array(range(n)).reshape(-1, 1)
    cls = DBSCAN(eps=threshold, min_samples=min_samples, metric=_metric).fit(samples)
    return cls.labels_
