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

    Examples::
        >>> from imgutils.validate import anime_furry_score
        >>>
        >>> anime_furry_score('non_furry/1.jpg')  # non-furry images
        {'non_furry': 0.9898804426193237, 'furry': 0.010119626298546791}
        >>> anime_furry_score('non_furry/2.jpg')
        {'non_furry': 0.9677742123603821, 'furry': 0.032225821167230606}
        >>> anime_furry_score('non_furry/3.jpg')
        {'non_furry': 0.959551215171814, 'furry': 0.040448784828186035}
        >>> anime_furry_score('non_furry/4.jpg')
        {'non_furry': 0.9535530209541321, 'furry': 0.04644693806767464}
        >>>
        >>> anime_furry_score('furry/5.jpg')  # furry images
        {'non_furry': 0.04358793422579765, 'furry': 0.9564120769500732}
        >>> anime_furry_score('furry/6.jpg')
        {'non_furry': 0.02767963521182537, 'furry': 0.9723203182220459}
        >>> anime_furry_score('furry/7.jpg')
        {'non_furry': 0.028900373727083206, 'furry': 0.9710996150970459}
        >>> anime_furry_score('furry/8.jpg')
        {'non_furry': 0.037573859095573425, 'furry': 0.9624261260032654}
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

    Examples::
        >>> from imgutils.validate import anime_furry
        >>>
        >>> anime_furry('non_furry/1.jpg')  # non-furry images
        ('non_furry', 0.9898804426193237)
        >>> anime_furry('non_furry/2.jpg')
        ('non_furry', 0.9677742123603821)
        >>> anime_furry('non_furry/3.jpg')
        ('non_furry', 0.959551215171814)
        >>> anime_furry('non_furry/4.jpg')
        ('non_furry', 0.9535530209541321)
        >>>
        >>> anime_furry('furry/5.jpg')  # furry images
        ('furry', 0.9564120769500732)
        >>> anime_furry('furry/6.jpg')
        ('furry', 0.9723203182220459)
        >>> anime_furry('furry/7.jpg')
        ('furry', 0.9710996150970459)
        >>> anime_furry('furry/8.jpg')
        ('furry', 0.9624261260032654)
    """
    return classify_predict(image, _REPO_ID, model_name)
