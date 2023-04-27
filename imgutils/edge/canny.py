from functools import partial
from typing import Optional

import cv2
import numpy as np

from ._base import _get_image_edge
from ..data import ImageTyping, load_image


def get_edge_by_canny(image: ImageTyping, low_threshold=100, high_threshold=200):
    image = load_image(image, mode='RGB')
    img = cv2.Canny(np.array(image), low_threshold, high_threshold)
    return img.astype(np.float32) / 255.0


def edge_image_with_canny(image: ImageTyping, low_threshold=100, high_threshold=200,
                          backcolor: str = 'white', forecolor: Optional[str] = None):
    return _get_image_edge(
        image,
        partial(get_edge_by_canny, low_threshold=low_threshold, high_threshold=high_threshold),
        backcolor, forecolor
    )
