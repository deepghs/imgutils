"""
Overview:
    Implementation of `PSNR <https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio>`_ metrics.
"""
import numpy as np

from ..data import rgb_encode

__all__ = [
    'psnr',
]


def psnr(img1, img2) -> float:
    """
    Overview:
        Psnr difference between images.

    :param img1: First image.
    :param img2: Second image.
    :return: Psnr difference of these two images.

    Example:
        Here are some images for example

        .. image:: psnr.plot.py.svg
           :align: center

        >>> from imgutils.metrics import psnr
        >>>
        >>> psnr('psnr/origin.jpg', 'psnr/origin.jpg')  # same image
        inf
        >>> psnr('psnr/origin.jpg', 'psnr/gaussian_20.dat.jpg')
        15.058228614646987
        >>> psnr('psnr/origin.jpg', 'psnr/gaussian_3.dat.jpg')
        27.65611098737784
        >>> psnr('psnr/origin.jpg', 'psnr/lq.dat.jpg')
        25.29589659377844
    """
    d1, d2 = rgb_encode(img1), rgb_encode(img2)
    mse = np.mean((d1 - d2) ** 2)
    return float(10 * np.log10(1. / mse))
