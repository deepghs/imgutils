"""
Overview:
    A model for rating anime images into 4 classes (``general``, ``sensitive``, ``questionable`` and ``explicit``),
    based on danbooru rating system.

    The following are sample images for testing.

    .. collapse:: The following are sample images for testing. (WARNING: NSFW!!!)

        .. image:: dbrating.plot.py.svg
            :align: center

    This is an overall benchmark of all the rating validation models:

    .. image:: dbrating_benchmark.plot.py.svg
        :align: center

    The models are hosted on
    `huggingface - deepghs/anime_dbrating <https://huggingface.co/deepghs/anime_dbrating>`_.

    .. note::
        This model is based on danbooru rating system, trained with 1.2 million images.
        If you need 3-level rating prediction, use :func:`imgutils.validate.rating.anime_rating`.

    .. note::
        Please note that the classification of ``general``, ``sensitive``, ``questionable`` and ``explicit`` types
        does not have clear boundaries, making it challenging to clean the training data. As a result,
        there is no strict ground truth for the rating classification problem. The judgment functionality
        provided by the current module is intended as a quick and rough estimation.

        **If you require an accurate filtering or judgment function specifically for R-18 images,
        it is recommended to consider using object detection-based methods**,
        such as using :func:`imgutils.detect.censor.detect_censors` to detect sensitive regions as the basis for judgment.
"""
from typing import Tuple, Dict

from ..data import ImageTyping
from ..generic import classify_predict, classify_predict_score

__all__ = [
    'anime_dbrating_score',
    'anime_dbrating',
]

_DEFAULT_MODEL_NAME = 'mobilenetv3_large_100_v0_ls0.2'
_REPO_ID = 'deepghs/anime_dbrating'


def anime_dbrating_score(image: ImageTyping, model_name: str = _DEFAULT_MODEL_NAME) -> Dict[str, float]:
    """
    Overview:
        Predict the rating of the given image, return the score with as a dict object.

    :param image: Image to rating.
    :param model_name: Model to use. Default is ``mobilenetv3_large_100_v0_ls0.2``.
        All available models are listed on the benchmark plot above.
        If you need better accuracy, just set this to ``caformer_s36_v0_ls0.2``.
    :return: A dict with ratings and scores.

    """
    return classify_predict_score(image, _REPO_ID, model_name)


def anime_dbrating(image: ImageTyping, model_name: str = _DEFAULT_MODEL_NAME) -> Tuple[str, float]:
    """
    Overview:
        Predict the rating of the given image, return the class and its score.

    :param image: Image to rating.
    :param model_name: Model to use. Default is ``mobilenetv3_large_100_v0_ls0.2``. All available models are listed
        on the benchmark plot above. If you need better accuracy, just set this to ``caformer_s36_v0_ls0.2``.
    :return: A tuple contains the rating and its score.


    """
    return classify_predict(image, _REPO_ID, model_name)
