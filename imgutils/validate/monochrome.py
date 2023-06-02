"""
Overview:
    A model for screening monochrome images, with the definition of monochrome images referring to Danbooru.

    The following are testing images. The top two rows are monochrome images, and the bottom two rows are color images.
    Please note that **monochrome images are not only those with all pixels in grayscale**.

    .. image:: monochrome.plot.py.svg
        :align: center

    This is an overall benchmark of all the monochrome validation models:

    .. image:: monochrome_benchmark.plot.py.svg
        :align: center
"""
from functools import lru_cache
from typing import Optional, Tuple, Mapping

import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

from ..data import ImageTyping, load_image, rgb_encode
from ..utils import open_onnx_model

__all__ = [
    'get_monochrome_score',
    'is_monochrome',
]

_MODELS: Mapping[Tuple[str, bool], str] = {
    ('caformer_s36', False): 'caformer_s36',
    ('mobilenetv3', False): 'mobilenetv3_large_100',
    ('mobilenetv3', True): 'mobilenetv3_large_100_safe2',
}


@lru_cache()
def _monochrome_validate_model(model: str, safe: bool):
    return open_onnx_model(hf_hub_download(
        f'deepghs/monochrome_detect',
        f'{_MODELS[(model, safe)]}/model.onnx',
    ))


def _2d_encode(image: Image.Image, size: Tuple[int, int] = (384, 384),
               normalize: Optional[Tuple[float, float]] = (0.5, 0.5)):
    image = image.resize(size, Image.BILINEAR)
    data = rgb_encode(image, order_='CHW')

    if normalize is not None:
        mean_, std_ = normalize
        mean = np.asarray([mean_]).reshape((-1, 1, 1))
        std = np.asarray([std_]).reshape((-1, 1, 1))
        data = (data - mean) / std

    return data


def get_monochrome_score(image: ImageTyping, model: str = 'mobilenetv3', safe: bool = True) -> float:
    """
    Overview:
        Get monochrome score of the given image.

    :param image: Image to predict, can be a ``PIL.Image`` object or the path of the image file.
    :param model: The model used for inference. The default value is ``mobilenetv3``,
        which offers high runtime performance.
    :param safe: Whether to enable the safe mode. When enabled, calculations will be performed using a model
        with higher precision but lower recall. The default value is ``True``.

    Examples::
        >>> import os
        >>> from imgutils.validate import get_monochrome_score
        >>>
        >>> get_monochrome_score('mono/1.jpg')  # monochrome images
        0.9789709448814392
        >>> get_monochrome_score('mono/2.jpg')
        0.973383903503418
        >>> get_monochrome_score('mono/3.jpg')
        0.9789378046989441
        >>> get_monochrome_score('mono/4.jpg')
        0.9920350909233093
        >>> get_monochrome_score('mono/5.jpg')
        0.9865685701370239
        >>> get_monochrome_score('mono/6.jpg')
        0.9589458703994751
        >>> get_monochrome_score('colored/7.jpg')  # colored images
        0.019315600395202637
        >>> get_monochrome_score('colored/8.jpg')
        0.008630834519863129
        >>> get_monochrome_score('colored/9.jpg')
        0.08635691553354263
        >>> get_monochrome_score('colored/10.jpg')
        0.01357574388384819
        >>> get_monochrome_score('colored/11.jpg')
        0.00710612116381526
        >>> get_monochrome_score('colored/12.jpg')
        0.025258518755435944
    """
    safe = bool(safe)
    if (model, safe) not in _MODELS:
        raise ValueError(f'Unknown model for monochrome detection - {model!r}, {safe!r}.')

    image = load_image(image, mode='RGB')
    input_data = _2d_encode(image).astype(np.float32)
    input_data = np.stack([input_data])
    output_data, = _monochrome_validate_model(model, safe).run(['output'], {'input': input_data})
    return output_data[0][0].item()


def is_monochrome(image: ImageTyping, threshold: float = 0.5,
                  model: str = 'mobilenetv3', safe: bool = True) -> bool:
    """
    Overview:
        Predict if the image is monochrome.

    :param image: Image to predict, can be a ``PIL.Image`` object or the path of the image file.
    :param threshold: Threshold value during prediction. If the score is higher than the threshold,
        the image will be classified as monochrome.
    :param model: The model used for inference. The default value is ``mobilenetv3``,
        which offers high runtime performance.
    :param safe: Safe level, with optional values including ``0``, ``2``, and ``4``,
        corresponding to different levels of the model. The default value is 2.
        For more technical details about this model, please refer to:
        https://huggingface.co/deepghs/imgutils-models#monochrome .

    Examples:
        >>> import os
        >>> from imgutils.validate import is_monochrome
        >>>
        >>> is_monochrome('mono/1.jpg')  # monochrome images
        True
        >>> is_monochrome('mono/2.jpg')
        True
        >>> is_monochrome('mono/3.jpg')
        True
        >>> is_monochrome('mono/4.jpg')
        True
        >>> is_monochrome('mono/5.jpg')
        True
        >>> is_monochrome('mono/6.jpg')
        True
        >>> is_monochrome('colored/7.jpg')  # colored images
        False
        >>> is_monochrome('colored/8.jpg')
        False
        >>> is_monochrome('colored/9.jpg')
        False
        >>> is_monochrome('colored/10.jpg')
        False
        >>> is_monochrome('colored/11.jpg')
        False
        >>> is_monochrome('colored/12.jpg')
        False
    """
    return get_monochrome_score(image, model, safe) >= threshold
