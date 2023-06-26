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

    """
    output = _raw_anime_rating(image, model_name)[0]
    max_id = np.argmax(output)
    return _open_anime_rating_labels(model_name)[max_id], output[max_id].item()
