"""
Overview:
    Get edge with ``cv2.Canny``.

    Having **the fastest running speed and the lowest system resource overhead**.
"""
from functools import partial
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from ._base import _get_image_edge
from ..data import ImageTyping, load_image


def get_edge_by_canny(image: ImageTyping, low_threshold=100, high_threshold=200):
    """
    Overview:
        Get edge mask with ``cv2.Canny``.

    :param image: Original image (assuming its size is ``HxW``).
    :param low_threshold: Low threshold of canny, default is ``100``.
    :param high_threshold: High threshold of canny, default is ``200``.
    :return: A mask with format ``float32[H, W]``.
    """
    image = load_image(image, mode='RGB')
    img = cv2.Canny(np.array(image), low_threshold, high_threshold)
    return img.astype(np.float32) / 255.0


def edge_image_with_canny(image: ImageTyping, low_threshold=100, high_threshold=200,
                          backcolor: str = 'white', forecolor: Optional[str] = None) -> Image.Image:
    """
    Overview:
        Get an image with the extracted edge from ``image``.

    :param image: Original image (assuming its size is ``HxW``).
    :param low_threshold: Low threshold of canny, default is ``100``.
    :param high_threshold: High threshold of canny, default is ``200``.
    :param backcolor: Background color the target image. Default is ``white``. When ``transparent`` is given, \
        the background will be transparent.
    :param forecolor: Fore color of the target image. Default is ``None`` which means use the color \
        from the given ``image``.
    :return: An image with the extracted edge from ``image``.

    Examples::
        .. image:: canny.plot.py.svg
            :align: center
    """
    return _get_image_edge(
        image,
        partial(get_edge_by_canny, low_threshold=low_threshold, high_threshold=high_threshold),
        backcolor, forecolor
    )
