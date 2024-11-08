"""
Overview:
    Get edge with lineart anime model.
"""
from functools import partial
from typing import Optional

import numpy as np
from huggingface_hub import hf_hub_download

from ._base import resize_image, cv2_resize, _get_image_edge
from ..data import ImageTyping, load_image
from ..utils import open_onnx_model, ts_lru_cache


def _preprocess(input_image, detect_resolution: int = 512):
    input_image = np.array(input_image, dtype=np.uint8)
    img = resize_image(input_image, detect_resolution, align=256).astype(np.float32)
    img = (img / 127.5 - 1.0).transpose(2, 0, 1)[None, ...]
    return img


@ts_lru_cache()
def _open_la_anime_model():
    return open_onnx_model(hf_hub_download(
        'deepghs/imgutils-models',
        f'lineart/lineart_anime.onnx',
    ))


def get_edge_by_lineart_anime(image: ImageTyping, detect_resolution: int = 512):
    """
    Overview:
        Get edge mask with lineart anime model.

    :param image: Original image (assuming its size is ``HxW``).
    :param detect_resolution: Resolution when passing the image into neural network. Default is ``512``.
    :return: A mask with format ``float32[H, W]``.
    """
    image = load_image(image, mode='RGB')
    output_, = _open_la_anime_model().run(['output'], {'input': _preprocess(image, detect_resolution)})
    output_ = (output_ + 1.0) / 2.0
    output_ = cv2_resize(output_[0].transpose(1, 2, 0), image.width, image.height)
    return 1.0 - output_.clip(0.0, 1.0)


def edge_image_with_lineart_anime(image: ImageTyping, detect_resolution: int = 512,
                                  backcolor: str = 'white', forecolor: Optional[str] = None):
    """
    Overview:
        Get an image with the extracted edge from ``image``.

    :param image: Original image (assuming its size is ``HxW``).
    :param detect_resolution: Resolution when passing the image into neural network. Default is ``512``.
    :param backcolor: Background color the target image. Default is ``white``. When ``transparent`` is given, \
        the background will be transparent.
    :param forecolor: Fore color of the target image. Default is ``None`` which means use the color \
        from the given ``image``.
    :return: An image with the extracted edge from ``image``.

    Examples::
        .. image:: lineart_anime.plot.py.svg
            :align: center
    """
    return _get_image_edge(
        image,
        partial(get_edge_by_lineart_anime, detect_resolution=detect_resolution),
        backcolor, forecolor
    )
