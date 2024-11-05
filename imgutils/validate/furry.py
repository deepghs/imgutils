"""
Overview:
    A model for classifying anime furry images into 2 classes (``non_furry``, ``furry``).

    The following are sample images for testing.

    .. image:: furry.plot.py.svg
        :align: center

    This is an overall benchmark of all the furry classification models:

    .. image:: furry_benchmark.plot.py.svg
        :align: center

    The models are hosted on
    `huggingface - deepghs/anime_furry <https://huggingface.co/deepghs/anime_furry>`_.
"""
from typing import Tuple, Dict

from ..data import ImageTyping
from ..generic import classify_predict, classify_predict_score

__all__ = [
    'anime_furry_score',
    'anime_furry',
]

_DEFAULT_MODEL_NAME = 'mobilenetv3_v0.1_dist'
_REPO_ID = 'deepghs/anime_furry'


def anime_furry_score(image: ImageTyping, model_name: str = _DEFAULT_MODEL_NAME) -> Dict[str, float]:
    """
    Get the scores for different types in a furry anime image.

    :param image: The input image.
    :type image: ImageTyping
    :param model_name: The model name. Default is 'mobilenetv3_v0.1_dist'.
    :type model_name: str
    :return: A dictionary with type scores.
    :rtype: Dict[str, float]
    """
    return classify_predict_score(image, _REPO_ID, model_name)


def anime_furry(image: ImageTyping, model_name: str = _DEFAULT_MODEL_NAME) -> Tuple[str, float]:
    """
    Get the primary furry type and its score.

    :param image: The input image.
    :type image: ImageTyping
    :param model_name: The model name. Default is 'mobilenetv3_v0.1_dist'.
    :type model_name: str
    :return: A tuple with the primary type and its score.
    :rtype: Tuple[str, float]
    """
    return classify_predict(image, _REPO_ID, model_name)
