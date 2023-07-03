"""
Overview:
    Tools for performing super-resolution processing on anime images based on the RealESRGAN model.

    This is an overall benchmark of all the operations in RealESRGAN models:

    .. image:: real_esrgan_benchmark.plot.py.svg
        :align: center
"""
import math
from functools import lru_cache
from typing import Tuple

import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

from imgutils.data import ImageTyping, load_image
from imgutils.utils import open_onnx_model

__all__ = [
    'real_esrgan_upscape_4x',
    'real_esrgan_upscale',
    'real_esrgan_resize',
]

_MODELS = [
    'RealESRGAN_x4plus_anime_4B32F',
    'RealESRGAN_x4plus_anime_6B',
]


@lru_cache()
def _open_real_esrgan_model(model: str):
    """
    Open the RealESRGAN model specified by the given model name.

    :param model: The name of the RealESRGAN model.
    :type model: str
    :returns: The opened RealESRGAN model.
    """
    return open_onnx_model(hf_hub_download(
        f'deepghs/imgutils-models',
        f'real_esrgan/{model}.onnx'
    ))


def _upscale_4x_for_rgb(array_rgb, model: str) -> Image:
    """
    Upscale the RGB image array 4 times using the specified RealESRGAN model.

    :param array_rgb: The RGB image array to upscale.
    :type array_rgb: np.ndarray
    :param model: The name of the RealESRGAN model.
    :type model: str
    :returns: The upscaled image.
    :rtype: Image
    """
    input_ = array_rgb[:, :, ::-1].astype(np.float32) / 255.0
    input_ = input_.transpose(2, 0, 1)[None, ...]

    output, = _open_real_esrgan_model(model).run(['image'], {"image.1": input_})
    output = (np.clip(output[0], a_min=0.0, a_max=1.0) * 255.0).round()
    output = output.transpose(1, 2, 0).astype(np.uint8)[:, :, ::-1]
    return output


def _upscale_4x_for_alpha(array_alpha, model: str) -> Image:
    """
    Upscale the alpha channel array 4 times using the specified RealESRGAN model.

    :param array_alpha: The alpha channel array to upscale.
    :type array_alpha: np.ndarray
    :param model: The name of the RealESRGAN model.
    :type model: str
    :returns: The upscaled alpha channel.
    :rtype: Image
    """
    array_rgb = np.stack([array_alpha, array_alpha, array_alpha]).transpose(1, 2, 0)
    upscaled_rgb = _upscale_4x_for_rgb(array_rgb, model)
    return upscaled_rgb[:, :, 0]


def real_esrgan_upscape_4x(image: ImageTyping, model: str = 'RealESRGAN_x4plus_anime_6B',
                           keep_alpha_for_rgba: bool = True) -> Image.Image:
    """
    Upscale the input image 4 times using the specified RealESRGAN model.

    :param image: The input image to upscale.
    :type image: ImageTyping
    :param model: The name of the RealESRGAN model. (default: 'RealESRGAN_x4plus_anime_6B')
    :type model: str
    :param keep_alpha_for_rgba: Whether to keep the alpha channel for RGBA images. (default: True)
    :type keep_alpha_for_rgba: bool
    :returns: The upscaled image.
    :rtype: Image.Image
    """
    img = load_image(image, force_background=None)
    if img.mode == 'RGBA' and keep_alpha_for_rgba:
        array = np.array(img)
        array_rgb, array_alpha = array[:, :, :3], array[:, :, 3]
        upscaled_rgb = _upscale_4x_for_rgb(array_rgb, model)
        upscaled_alpha = _upscale_4x_for_alpha(array_alpha, model)
        data = np.zeros((upscaled_rgb.shape[0], upscaled_rgb.shape[1], 4), dtype=np.uint8)
        data[:, :, :3] = upscaled_rgb
        data[:, :, 3] = upscaled_alpha
        return Image.fromarray(data, mode='RGBA')

    else:
        img = load_image(img, mode='RGB')
        return Image.fromarray(_upscale_4x_for_rgb(np.array(img), model), mode='RGB')


def real_esrgan_upscale(image: ImageTyping, ratio: float, model: str = 'RealESRGAN_x4plus_anime_6B',
                        keep_alpha_for_rgba: bool = True) -> Image.Image:
    """
    Upscale the input image by the specified ratio using the specified RealESRGAN model.

    :param image: The input image to upscale.
    :type image: ImageTyping
    :param ratio: The upscale ratio.
    :type ratio: float
    :param model: The name of the RealESRGAN model. (default: 'RealESRGAN_x4plus_anime_6B')
    :type model: str
    :param keep_alpha_for_rgba: Whether to keep the alpha channel for RGBA images. (default: True)
    :type keep_alpha_for_rgba: bool
    :returns: The upscaled image.
    :rtype: Image.Image
    """
    image = load_image(image, force_background=None)
    cnt_4x = 0
    while ratio > 1:
        cnt_4x += 1
        ratio /= 4.0

    for i in range(cnt_4x):
        image = real_esrgan_upscape_4x(image, model, keep_alpha_for_rgba)

    new_size = (int(math.ceil(image.width * ratio)), int(math.ceil(image.height * ratio)))
    return image.resize(new_size)


def real_esrgan_resize(image: ImageTyping, size: Tuple[int, int], model: str = 'RealESRGAN_x4plus_anime_6B',
                       keep_alpha_for_rgba: bool = True) -> Image.Image:
    """
    Resize the input image to the specified size using the specified RealESRGAN model.

    :param image: The input image to resize.
    :type image: ImageTyping
    :param size: The target size of the image (width, height).
    :type size: Tuple[int, int]
    :param model: The name of the RealESRGAN model. (default: 'RealESRGAN_x4plus_anime_6B')
    :type model: str
    :param keep_alpha_for_rgba: Whether to keep the alpha channel for RGBA images. (default: True)
    :type keep_alpha_for_rgba: bool
    :returns: The resized image.
    :rtype: Image.Image
    """
    image = load_image(image, force_background=None)
    width, height = image.size
    twidth, theight = size
    ratio = 4.0 ** math.ceil(math.log(max(twidth / width, theight / height), 4))
    image = real_esrgan_upscale(image, ratio, model, keep_alpha_for_rgba)
    return image.resize(size)
