"""
Overview:
    A model for classifying anime portrait images into 3 classes (``person``, ``halfbody``, ``head``).

    The following are sample images for testing.

    .. image:: portrait.plot.py.svg
        :align: center

    This is an overall benchmark of all the portrait classification models:

    .. image:: portrait_benchmark.plot.py.svg
        :align: center

    The models are hosted on
    `huggingface - deepghs/anime_portrait <https://huggingface.co/deepghs/anime_portrait>`_.
"""
import json
from functools import lru_cache
from typing import Tuple, Optional, Dict, List

import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

from imgutils.data import rgb_encode, ImageTyping, load_image
from imgutils.utils import open_onnx_model

__all__ = [
    'anime_portrait_score',
    'anime_portrait',
]

_DEFAULT_MODEL_NAME = 'mobilenetv3_v0_dist'


@lru_cache()
def _open_anime_portrait_model(model_name):
    """
    Open the anime portrait model.

    :param model_name: The model name.
    :type model_name: str
    :return: The ONNX model.
    """
    return open_onnx_model(hf_hub_download(
        f'deepghs/anime_portrait',
        f'{model_name}/model.onnx',
    ))


@lru_cache()
def _get_anime_portrait_labels(model_name) -> List[str]:
    """
    Get the labels for the anime portrait model.

    :param model_name: The model name.
    :type model_name: str
    :return: The list of labels.
    :rtype: List[str]
    """
    with open(hf_hub_download(
            f'deepghs/anime_portrait',
            f'{model_name}/meta.json',
    ), 'r') as f:
        return json.load(f)['labels']


def _img_encode(image: Image.Image, size: Tuple[int, int] = (384, 384),
                normalize: Optional[Tuple[float, float]] = (0.5, 0.5)):
    """
    Encode the input image.

    :param image: The input image.
    :type image: Image.Image
    :param size: The desired size of the image.
    :type size: Tuple[int, int]
    :param normalize: Mean and standard deviation for normalization. Default is (0.5, 0.5).
    :type normalize: Optional[Tuple[float, float]]
    :return: The encoded image data.
    :rtype: np.ndarray
    """
    image = image.resize(size, Image.BILINEAR)
    data = rgb_encode(image, order_='CHW')

    if normalize is not None:
        mean_, std_ = normalize
        mean = np.asarray([mean_]).reshape((-1, 1, 1))
        std = np.asarray([std_]).reshape((-1, 1, 1))
        data = (data - mean) / std

    return data.astype(np.float32)


def _raw_anime_portrait(image: ImageTyping, model_name: str = _DEFAULT_MODEL_NAME):
    """
    Perform raw anime portrait processing on the input image.

    :param image: The input image.
    :type image: ImageTyping
    :param model_name: The model name. Default is 'mobilenetv3_v0_dist'.
    :type model_name: str
    :return: The processed image data.
    :rtype: np.ndarray
    """
    image = load_image(image, force_background='white', mode='RGB')
    input_ = _img_encode(image)[None, ...]
    output, = _open_anime_portrait_model(model_name).run(['output'], {'input': input_})
    return output


def anime_portrait_score(image: ImageTyping, model_name: str = _DEFAULT_MODEL_NAME) -> Dict[str, float]:
    """
    Get the scores for different types in an anime portrait.

    :param image: The input image.
    :type image: ImageTyping
    :param model_name: The model name. Default is 'mobilenetv3_v0_dist'.
    :type model_name: str
    :return: A dictionary with type scores.
    :rtype: Dict[str, float]

    Examples::
        >>> from imgutils.validate import anime_portrait_score
        >>>
        >>> anime_portrait_score('person/1.jpg')
        {'person': 0.999900221824646, 'halfbody': 8.645313209854066e-05, 'head': 1.3387104445428122e-05}
        >>> anime_portrait_score('person/2.jpg')
        {'person': 0.9999704360961914, 'halfbody': 2.4465465685352683e-05, 'head': 5.071506166132167e-06}
        >>> anime_portrait_score('person/3.jpg')
        {'person': 0.9999785423278809, 'halfbody': 1.512719154561637e-05, 'head': 6.292278612818336e-06}
        >>> anime_portrait_score('halfbody/4.jpg')
        {'person': 4.919455750496127e-05, 'halfbody': 0.9999444484710693, 'head': 6.3647335082350764e-06}
        >>> anime_portrait_score('halfbody/5.jpg')
        {'person': 1.0555699191172607e-05, 'halfbody': 0.9999880790710449, 'head': 1.3210242286731955e-06}
        >>> anime_portrait_score('halfbody/6.jpg')
        {'person': 1.7451418898417614e-05, 'halfbody': 0.9999822378158569, 'head': 3.2084267331811134e-07}
        >>> anime_portrait_score('head/7.jpg')
        {'person': 2.7460413321023225e-07, 'halfbody': 1.1532473820352607e-07, 'head': 0.9999996423721313}
        >>> anime_portrait_score('head/8.jpg')
        {'person': 1.0316136922483565e-07, 'halfbody': 5.840229633236049e-08, 'head': 0.9999998807907104}
        >>> anime_portrait_score('head/9.jpg')
        {'person': 5.736660568800289e-07, 'halfbody': 7.199210472208506e-08, 'head': 0.9999992847442627}
    """
    output = _raw_anime_portrait(image, model_name)
    values = dict(zip(_get_anime_portrait_labels(model_name), map(lambda x: x.item(), output[0])))
    return values


def anime_portrait(image: ImageTyping, model_name: str = _DEFAULT_MODEL_NAME) -> Tuple[str, float]:
    """
    Get the primary anime portrait type and its score.

    :param image: The input image.
    :type image: ImageTyping
    :param model_name: The model name. Default is 'mobilenetv3_v0_dist'.
    :type model_name: str
    :return: A tuple with the primary type and its score.
    :rtype: Tuple[str, float]

    Examples::
        >>> from imgutils.validate import anime_portrait
        >>>
        >>> anime_portrait('person/1.jpg')
        ('person', 0.999900221824646)
        >>> anime_portrait('person/2.jpg')
        ('person', 0.9999704360961914)
        >>> anime_portrait('person/3.jpg')
        ('person', 0.9999785423278809)
        >>> anime_portrait('halfbody/4.jpg')
        ('halfbody', 0.9999444484710693)
        >>> anime_portrait('halfbody/5.jpg')
        ('halfbody', 0.9999880790710449)
        >>> anime_portrait('halfbody/6.jpg')
        ('halfbody', 0.9999822378158569)
        >>> anime_portrait('head/7.jpg')
        ('head', 0.9999996423721313)
        >>> anime_portrait('head/8.jpg')
        ('head', 0.9999998807907104)
        >>> anime_portrait('head/9.jpg')
        ('head', 0.9999992847442627)
    """
    output = _raw_anime_portrait(image, model_name)[0]
    max_id = np.argmax(output)
    return _get_anime_portrait_labels(model_name)[max_id], output[max_id].item()
