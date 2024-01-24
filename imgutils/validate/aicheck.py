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
from ..data import ImageTyping
from ..generic import classify_predict, classify_predict_score

__all__ = [
    'get_ai_created_score',
    'is_ai_created',
]

_DEFAULT_MODEL_NAME = 'mobilenetv3_sce_dist'
_REPO_ID = 'deepghs/anime_ai_check'


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
    return classify_predict_score(image, _REPO_ID, model_name)['ai']


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
    type_, _ = classify_predict(image, _REPO_ID, model_name)
    return type_ == 'ai'
