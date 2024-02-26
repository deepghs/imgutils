"""
Overview:
    Rough Prediction of illustration's completeness (``monochrome``, ``rough`` and ``polished``).

    The following are sample images for testing.

    .. image:: completeness.plot.py.svg
        :align: center

    This is an overall benchmark of all the completeness classification models:

    .. image:: completeness_benchmark.plot.py.svg
        :align: center

    The models are hosted on
    `huggingface - deepghs/anime_completeness <https://huggingface.co/deepghs/anime_completeness>`_.
"""
from typing import Tuple, Dict

from ..data import ImageTyping
from ..generic import classify_predict, classify_predict_score

__all__ = [
    'anime_completeness_score',
    'anime_completeness',
]

_DEFAULT_MODEL_NAME = 'mobilenetv3_v2.2_dist'
_REPO_ID = 'deepghs/anime_completeness'


def anime_completeness_score(image: ImageTyping, model_name: str = _DEFAULT_MODEL_NAME) -> Dict[str, float]:
    """
    Rough Prediction of illustration's completeness, and return the score.

    :param image: The input image.
    :type image: ImageTyping
    :param model_name: The model name. Default is 'mobilenetv3_v2.2_dist'.
    :type model_name: str
    :return: A dictionary with type scores.
    :rtype: Dict[str, float]

    Examples::
    """
    return classify_predict_score(image, _REPO_ID, model_name)


def anime_completeness(image: ImageTyping, model_name: str = _DEFAULT_MODEL_NAME) -> Tuple[str, float]:
    """
    Rough Prediction of illustration's completeness, return the predict result and its score.

    :param image: The input image.
    :type image: ImageTyping
    :param model_name: The model name. Default is 'mobilenetv3_v2.2_dist'.
    :type model_name: str
    :return: A tuple with the primary type and its score.
    :rtype: Tuple[str, float]

    Examples::
    """
    return classify_predict(image, _REPO_ID, model_name)
