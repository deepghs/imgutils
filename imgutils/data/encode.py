import numpy as np

from .image import load_image, ImageTyping

_DEFAULT_ORDER = 'HWC'


def _get_hwc_map(order_: str):
    return tuple(_DEFAULT_ORDER.index(c) for c in order_.upper())


def rgb_encode(image: ImageTyping, order_: str = 'CHW', use_float: bool = True) -> np.ndarray:
    image = load_image(image, mode='RGB')
    array = np.asarray(image)
    array = np.transpose(array, _get_hwc_map(order_))
    if use_float:
        array = (array / 255.0).astype(np.float32)
        assert array.dtype == np.float32
    else:
        assert array.dtype == np.uint8
    return array
