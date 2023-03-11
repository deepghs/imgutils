import numpy as np

from ..data import rgb_encode

__all__ = [
    'psnr',
]


def psnr(img1, img2) -> float:
    d1, d2 = rgb_encode(img1), rgb_encode(img2)
    mse = np.mean((d1 - d2) ** 2)
    return float(10 * np.log10(1. / mse))
