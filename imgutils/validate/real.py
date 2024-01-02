"""
Overview:
    A model for classifying anime real images into 2 classes (``anime``, ``real``).

    The following are sample images for testing.

    .. image:: real.plot.py.svg
        :align: center

    This is an overall benchmark of all the real classification models:

    .. image:: real_benchmark.plot.py.svg
        :align: center

    The models are hosted on
    `huggingface - deepghs/anime_real_cls <https://huggingface.co/deepghs/anime_real_cls>`_.
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
    'anime_real_score',
    'anime_real',
]

_DEFAULT_MODEL_NAME = 'mobilenetv3_v0_dist'


@lru_cache()
def _open_anime_real_model(model_name):
    """
    Open the anime real model.

    :param model_name: The model name.
    :type model_name: str
    :return: The ONNX model.
    """
    return open_onnx_model(hf_hub_download(
        f'deepghs/anime_real_cls',
        f'{model_name}/model.onnx',
    ))


@lru_cache()
def _get_anime_real_labels(model_name) -> List[str]:
    """
    Get the labels for the anime real model.

    :param model_name: The model name.
    :type model_name: str
    :return: The list of labels.
    :rtype: List[str]
    """
    with open(hf_hub_download(
            f'deepghs/anime_real_cls',
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


def _raw_anime_real(image: ImageTyping, model_name: str = _DEFAULT_MODEL_NAME):
    """
    Perform raw anime real processing on the input image.

    :param image: The input image.
    :type image: ImageTyping
    :param model_name: The model name. Default is 'mobilenetv3_v0_dist'.
    :type model_name: str
    :return: The processed image data.
    :rtype: np.ndarray
    """
    image = load_image(image, force_background='white', mode='RGB')
    input_ = _img_encode(image)[None, ...]
    output, = _open_anime_real_model(model_name).run(['output'], {'input': input_})
    return output


def anime_real_score(image: ImageTyping, model_name: str = _DEFAULT_MODEL_NAME) -> Dict[str, float]:
    """
    Get the scores for different types in an anime real.

    :param image: The input image.
    :type image: ImageTyping
    :param model_name: The model name. Default is 'mobilenetv3_v0_dist'.
    :type model_name: str
    :return: A dictionary with type scores.
    :rtype: Dict[str, float]

    Examples::
        >>> from imgutils.validate import anime_real_score
        >>>
        >>> anime_real_score('real/anime/1.jpg')
        {'anime': 0.9999716281890869, 'real': 2.8398366339388303e-05}
        >>> anime_real_score('real/anime/2.jpg')
        {'anime': 0.9992202520370483, 'real': 0.0007797438884153962}
        >>> anime_real_score('real/anime/3.jpg')
        {'anime': 0.9999709129333496, 'real': 2.905452492996119e-05}
        >>> anime_real_score('real/anime/4.jpg')
        {'anime': 0.9999765157699585, 'real': 2.3499671442550607e-05}
        >>> anime_real_score('real/anime/5.jpg')
        {'anime': 0.9994087219238281, 'real': 0.0005913018831051886}
        >>> anime_real_score('real/anime/6.jpg')
        {'anime': 0.9999759197235107, 'real': 2.4061362637439743e-05}
        >>> anime_real_score('real/anime/7.jpg')
        {'anime': 0.9999052286148071, 'real': 9.475799015490338e-05}
        >>> anime_real_score('real/anime/8.jpg')
        {'anime': 0.9999759197235107, 'real': 2.403173675702419e-05}
        >>> anime_real_score('real/real/9.jpg')
        {'anime': 1.5848207794988411e-06, 'real': 0.9999984502792358}
        >>> anime_real_score('real/real/10.jpg')
        {'anime': 0.0010207017185166478, 'real': 0.9989792704582214}
        >>> anime_real_score('real/real/11.jpg')
        {'anime': 2.2124368115328252e-06, 'real': 0.9999977350234985}
        >>> anime_real_score('real/real/12.jpg')
        {'anime': 1.6512358342879452e-05, 'real': 0.9999834299087524}
        >>> anime_real_score('real/real/13.jpg')
        {'anime': 6.359853614412714e-06, 'real': 0.9999936819076538}
        >>> anime_real_score('real/real/14.jpg')
        {'anime': 1.600314317329321e-05, 'real': 0.9999840259552002}
        >>> anime_real_score('real/real/15.jpg')
        {'anime': 1.5589323083986528e-05, 'real': 0.9999843835830688}
        >>> anime_real_score('real/real/16.jpg')
        {'anime': 1.5513256585109048e-05, 'real': 0.9999845027923584}
    """
    output = _raw_anime_real(image, model_name)
    values = dict(zip(_get_anime_real_labels(model_name), map(lambda x: x.item(), output[0])))
    return values


def anime_real(image: ImageTyping, model_name: str = _DEFAULT_MODEL_NAME) -> Tuple[str, float]:
    """
    Get the primary anime real type and its score.

    :param image: The input image.
    :type image: ImageTyping
    :param model_name: The model name. Default is 'mobilenetv3_v0_dist'.
    :type model_name: str
    :return: A tuple with the primary type and its score.
    :rtype: Tuple[str, float]

    Examples::
        >>> from imgutils.validate import anime_real
        >>>
        >>> anime_real('real/anime/1.jpg')
        ('anime', 0.9999716281890869)
        >>> anime_real('real/anime/2.jpg')
        ('anime', 0.9992202520370483)
        >>> anime_real('real/anime/3.jpg')
        ('anime', 0.9999709129333496)
        >>> anime_real('real/anime/4.jpg')
        ('anime', 0.9999765157699585)
        >>> anime_real('real/anime/5.jpg')
        ('anime', 0.9994087219238281)
        >>> anime_real('real/anime/6.jpg')
        ('anime', 0.9999759197235107)
        >>> anime_real('real/anime/7.jpg')
        ('anime', 0.9999052286148071)
        >>> anime_real('real/anime/8.jpg')
        ('anime', 0.9999759197235107)
        >>> anime_real('real/real/9.jpg')
        ('real', 0.9999984502792358)
        >>> anime_real('real/real/10.jpg')
        ('real', 0.9989792704582214)
        >>> anime_real('real/real/11.jpg')
        ('real', 0.9999977350234985)
        >>> anime_real('real/real/12.jpg')
        ('real', 0.9999834299087524)
        >>> anime_real('real/real/13.jpg')
        ('real', 0.9999936819076538)
        >>> anime_real('real/real/14.jpg')
        ('real', 0.9999840259552002)
        >>> anime_real('real/real/15.jpg')
        ('real', 0.9999843835830688)
        >>> anime_real('real/real/16.jpg')
        ('real', 0.9999845027923584)
    """
    output = _raw_anime_real(image, model_name)[0]
    max_id = np.argmax(output)
    return _get_anime_real_labels(model_name)[max_id], output[max_id].item()
