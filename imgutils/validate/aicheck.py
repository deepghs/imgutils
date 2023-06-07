"""
Overview:
    A model for detecting AI-created images.

    The following are sample images for testing.

    .. image:: aicheck.plot.py.svg
        :align: center

    This is an overall benchmark of all the AI-check validation models:

    .. image:: aicheck_benchmark.plot.py.svg
        :align: center

    The models are hosted on
    `huggingface - deepghs/anime_ai_check <https://huggingface.co/deepghs/anime_ai_check>`_.
"""
from functools import lru_cache
from typing import Tuple, Optional

import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

from imgutils.data import rgb_encode, ImageTyping, load_image
from imgutils.utils import open_onnx_model

__all__ = [
    'get_ai_created_score',
    'is_ai_created',
]

_LABELS = ['ai', 'human']
_MODEL_NAMES = [
    'caformer_s36_plus_sce',
    'mobilenetv3_sce',
    'mobilenetv3_sce_dist',
]
_DEFAULT_MODEL_NAME = 'mobilenetv3_sce_dist'


@lru_cache()
def _open_anime_aicheck_model(model_name):
    return open_onnx_model(hf_hub_download(
        f'deepghs/anime_ai_check',
        f'{model_name}/model.onnx',
    ))


def _img_encode(image: Image.Image, size: Tuple[int, int] = (384, 384),
                normalize: Optional[Tuple[float, float]] = (0.5, 0.5)):
    image = image.resize(size, Image.BILINEAR)
    data = rgb_encode(image, order_='CHW')

    if normalize is not None:
        mean_, std_ = normalize
        mean = np.asarray([mean_]).reshape((-1, 1, 1))
        std = np.asarray([std_]).reshape((-1, 1, 1))
        data = (data - mean) / std

    return data.astype(np.float32)


def get_ai_created_score(image: ImageTyping, model_name: str = _DEFAULT_MODEL_NAME) -> float:
    """
    Overview:
        Predict if the given image is created by AI (mainly by stable diffusion), given a score.

    :param image: Image to be predicted.
    :param model_name: Name of the model. Default is ``mobilenetv3_sce_dist``.
        If you need better accuracy, use ``caformer_s36_plus_sce``.
        All the available values are listed on the benchmark graph.
    :return: A float number which represent the score of AI-check.

    Examples::
        >>> from imgutils.validate import is_ai_created
        >>>
        >>> is_ai_created('aicheck/ai/1.jpg')
        True
        >>> is_ai_created('aicheck/ai/2.jpg')
        True
        >>> is_ai_created('aicheck/ai/3.jpg')
        True
        >>> is_ai_created('aicheck/ai/4.jpg')
        True
        >>> is_ai_created('aicheck/ai/5.jpg')
        True
        >>> is_ai_created('aicheck/ai/6.jpg')
        True
        >>> is_ai_created('aicheck/human/7.jpg')
        False
        >>> is_ai_created('aicheck/human/8.jpg')
        False
        >>> is_ai_created('aicheck/human/9.jpg')
        False
        >>> is_ai_created('aicheck/human/10.jpg')
        False
        >>> is_ai_created('aicheck/human/11.jpg')
        False
        >>> is_ai_created('aicheck/human/12.jpg')
        False
    """
    image = load_image(image, force_background='white', mode='RGB')
    input_ = _img_encode(image)[None, ...]
    output, = _open_anime_aicheck_model(model_name).run(['output'], {'input': input_})

    return output[0][0].item()


def is_ai_created(image: ImageTyping, model_name: str = _DEFAULT_MODEL_NAME, threshold: float = 0.5) -> bool:
    """
    Overview:
        Predict if the given image is created by AI (mainly by stable diffusion).

    :param image: Image to be predicted.
    :param model_name: Name of the model. Default is ``mobilenetv3_sce_dist``.
        If you need better accuracy, use ``caformer_s36_plus_sce``.
        All the available values are listed on the benchmark graph.
    :param threshold: Threshold of the score. When the score is no less than ``threshold``, this image
        will be predicted as ``AI-created``. Default is ``0.5``.
    :return: This image is ``AI-created`` or not.

    Examples::
        >>> from imgutils.validate import get_ai_created_score
        >>>
        >>> get_ai_created_score('aicheck/ai/1.jpg')
        0.9996960163116455
        >>> get_ai_created_score('aicheck/ai/2.jpg')
        0.9999125003814697
        >>> get_ai_created_score('aicheck/ai/3.jpg')
        0.997803270816803
        >>> get_ai_created_score('aicheck/ai/4.jpg')
        0.9960069060325623
        >>> get_ai_created_score('aicheck/ai/5.jpg')
        0.9887709021568298
        >>> get_ai_created_score('aicheck/ai/6.jpg')
        0.9998629093170166
        >>> get_ai_created_score('aicheck/human/7.jpg')
        0.0013722758740186691
        >>> get_ai_created_score('aicheck/human/8.jpg')
        0.00020673229300882667
        >>> get_ai_created_score('aicheck/human/9.jpg')
        0.0001895089662866667
        >>> get_ai_created_score('aicheck/human/10.jpg')
        0.0008857478387653828
        >>> get_ai_created_score('aicheck/human/11.jpg')
        4.552320024231449e-05
        >>> get_ai_created_score('aicheck/human/12.jpg')
        0.001168627175502479
    """
    return get_ai_created_score(image, model_name) >= threshold
