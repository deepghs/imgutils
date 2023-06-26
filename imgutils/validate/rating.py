"""
Overview:
    A model for rating anime images into 4 classes (``safe``, ``r15`` and ``r18``).

    The following are sample images for testing.

    .. image:: rating.plot.py.svg
        :align: center

    This is an overall benchmark of all the rating validation models:

    .. image:: rating_benchmark.plot.py.svg
        :align: center

    The models are hosted on
    `huggingface - deepghs/anime_rating <https://huggingface.co/deepghs/anime_rating>`_.

    .. note::
        Please note that the classification of ``safe``, ``r15``, and ``r18`` types does not have clear boundaries,
        making it challenging to clean the training data. As a result, there is no strict ground truth
        for the rating classification problem. The judgment functionality provided by the current module
        is intended as a quick and rough estimation.

        **If you require an accurate filtering or judgment function specifically for R-18 images,
        it is recommended to consider using object detection-based methods**,
        such as using :func:`imgutils.detect.censor.detect_censors` to detect sensitive regions as the basis for judgment.
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
    'anime_rating_score',
    'anime_rating',
]

_MODEL_NAMES = [
    'caformer_s36_plus',
    'mobilenetv3',
    'mobilenetv3_sce',
    'mobilenetv3_sce_dist',
]
_DEFAULT_MODEL_NAME = 'mobilenetv3_sce_dist'


@lru_cache()
def _open_anime_rating_model(model_name):
    return open_onnx_model(hf_hub_download(
        f'deepghs/anime_rating',
        f'{model_name}/model.onnx',
    ))


@lru_cache()
def _open_anime_rating_labels(model_name) -> List[str]:
    with open(hf_hub_download(
            f'deepghs/anime_rating',
            f'{model_name}/meta.json',
    ), 'r') as f:
        return json.load(f)['labels']


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


def _raw_anime_rating(image: ImageTyping, model_name: str = _DEFAULT_MODEL_NAME):
    image = load_image(image, force_background='white', mode='RGB')
    input_ = _img_encode(image)[None, ...]
    output, = _open_anime_rating_model(model_name).run(['output'], {'input': input_})

    return output


def anime_rating_score(image: ImageTyping, model_name: str = _DEFAULT_MODEL_NAME) -> Dict[str, float]:
    """
    Overview:
        Predict the rating of the given image, return the score with as a dict object.

    :param image: Image to rating.
    :param model_name: Model to use. Default is ``mobilenetv3_sce_dist``. All available models are listed
        on the benchmark plot above. If you need better accuracy, just set this to ``caformer_s36_plus``.
    :return: A dict with ratings and scores.

    Example::
        >>> from imgutils.validate import anime_rating_score
        >>>
        >>> anime_rating_score('rating/safe/1.jpg')
        {'safe': 0.9999998807907104, 'r15': 2.5863172936624323e-08, 'r18': 6.480062353375615e-08}
        >>> anime_rating_score('rating/safe/2.jpg')
        {'safe': 0.9924363493919373, 'r15': 0.007255776319652796, 'r18': 0.0003077814180869609}
        >>> anime_rating_score('rating/safe/3.jpg')
        {'safe': 0.996969997882843, 'r15': 0.0030054834205657244, 'r18': 2.4601260520284995e-05}
        >>> anime_rating_score('rating/safe/4.jpg')
        {'safe': 0.9966891407966614, 'r15': 0.003293127752840519, 'r18': 1.770909148035571e-05}
        >>> anime_rating_score('rating/r15/5.jpg')
        {'safe': 0.00025384966284036636, 'r15': 0.9996721744537354, 'r18': 7.399192691082135e-05}
        >>> anime_rating_score('rating/r15/6.jpg')
        {'safe': 7.973351603141055e-05, 'r15': 0.9998563528060913, 'r18': 6.391309580067173e-05}
        >>> anime_rating_score('rating/r15/7.jpg')
        {'safe': 0.0018681309884414077, 'r15': 0.9827859997749329, 'r18': 0.015345841646194458}
        >>> anime_rating_score('rating/r15/8.jpg')
        {'safe': 0.013710384257137775, 'r15': 0.8339558839797974, 'r18': 0.15233369171619415}
        >>> anime_rating_score('rating/r18/9.jpg')
        {'safe': 3.951323833462084e-06, 'r15': 0.00029566374723799527, 'r18': 0.9997004270553589}
        >>> anime_rating_score('rating/r18/10.jpg')
        {'safe': 0.00018434497178532183, 'r15': 4.568440272123553e-05, 'r18': 0.9997699856758118}
        >>> anime_rating_score('rating/r18/11.jpg')
        {'safe': 9.11225129129889e-07, 'r15': 5.051862899563275e-05, 'r18': 0.9999485015869141}
        >>> anime_rating_score('rating/r18/12.jpg')
        {'safe': 6.902020231791539e-06, 'r15': 0.0005639699520543218, 'r18': 0.9994290471076965}
    """
    output = _raw_anime_rating(image, model_name)
    values = dict(zip(_open_anime_rating_labels(model_name), map(lambda x: x.item(), output[0])))
    return values


def anime_rating(image: ImageTyping, model_name: str = _DEFAULT_MODEL_NAME) -> Tuple[str, float]:
    """
    Overview:
        Predict the rating of the given image, return the class and its score.

    :param image: Image to rating.
    :param model_name: Model to use. Default is ``mobilenetv3_sce_dist``. All available models are listed
        on the benchmark plot above. If you need better accuracy, just set this to ``caformer_s36_plus``.
    :return: A tuple contains the rating and its score.

    Examples::
        >>> from imgutils.validate import anime_rating
        >>>
        >>> anime_rating('rating/safe/1.jpg')
        ('safe', 0.9999998807907104)
        >>> anime_rating('rating/safe/2.jpg')
        ('safe', 0.9924363493919373)
        >>> anime_rating('rating/safe/3.jpg')
        ('safe', 0.996969997882843)
        >>> anime_rating('rating/safe/4.jpg')
        ('safe', 0.9966891407966614)
        >>> anime_rating('rating/r15/5.jpg')
        ('r15', 0.9996721744537354)
        >>> anime_rating('rating/r15/6.jpg')
        ('r15', 0.9998563528060913)
        >>> anime_rating('rating/r15/7.jpg')
        ('r15', 0.9827859997749329)
        >>> anime_rating('rating/r15/8.jpg')
        ('r15', 0.8339558839797974)
        >>> anime_rating('rating/r18/9.jpg')
        ('r18', 0.9997004270553589)
        >>> anime_rating('rating/r18/10.jpg')
        ('r18', 0.9997699856758118)
        >>> anime_rating('rating/r18/11.jpg')
        ('r18', 0.9999485015869141)
        >>> anime_rating('rating/r18/12.jpg')
        ('r18', 0.9994290471076965)
    """
    output = _raw_anime_rating(image, model_name)[0]
    max_id = np.argmax(output)
    return _open_anime_rating_labels(model_name)[max_id], output[max_id].item()
