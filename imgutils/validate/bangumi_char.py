"""
Overview:
    A model for classifying anime bangumi character images into 4 classes
    (``vision``, ``imagery``, ``halfbody``, ``face``).

    The following are sample images for testing.

    .. image:: bangumi_char.plot.py.svg
        :align: center

    This is an overall benchmark of all the bangumi character classification models:

    .. image:: bangumi_char_benchmark.plot.py.svg
        :align: center

    The models are hosted on
    `huggingface - deepghs/bangumi_char_type <https://huggingface.co/deepghs/bangumi_char_type>`_.

    .. note::
        Please note that the classification of bangumi character types is not based on the proportion
        of the head in the image but on the completeness of facial details.
        The specific definitions of the four types can be found `here <https://huggingface.co/datasets/deepghs/bangumi_char_type>`_.

        In anime videos, **characters in secondary positions often lack details due to simplified animation**,
        leading to their classification under the ``vision`` category.
        **The other three types include images with complete and clear facial features**.

        If you are looking for a classification model that judges the proportion of the head in an image,
        please use the :func:`imgutils.validate.anime_portrait` function.
"""
from typing import Tuple, Dict

from ..data import ImageTyping
from ..generic import classify_predict_score, classify_predict

__all__ = [
    'anime_bangumi_char_score',
    'anime_bangumi_char',
]

_DEFAULT_MODEL_NAME = 'mobilenetv3_v0_dist'
_REPO_ID = 'deepghs/bangumi_char_type'


def anime_bangumi_char_score(image: ImageTyping, model_name: str = _DEFAULT_MODEL_NAME) -> Dict[str, float]:
    """
    Get the scores for different types in an anime bangumi character.

    :param image: The input image.
    :type image: ImageTyping
    :param model_name: The model name. Default is 'mobilenetv3_v0_dist'.
    :type model_name: str
    :return: A dictionary with type scores.
    :rtype: Dict[str, float]

    Examples::
        >>> from imgutils.validate import anime_bangumi_char_score
        >>>
        >>> anime_bangumi_char_score('bangumi_char/vision/1.jpg')
        {'vision': 0.9998525381088257, 'imagery': 0.00012103465269319713, 'halfbody': 1.6464786313008517e-05, 'face': 9.906112609314732e-06}
        >>> anime_bangumi_char_score('bangumi_char/vision/2.jpg')
        {'vision': 0.9997243285179138, 'imagery': 0.0002490800397936255, 'halfbody': 1.7215803381986916e-05, 'face': 9.354368557978887e-06}
        >>> anime_bangumi_char_score('bangumi_char/vision/3.jpg')
        {'vision': 0.9998849630355835, 'imagery': 8.90006631379947e-05, 'halfbody': 1.3920385754317977e-05, 'face': 1.2084233276254963e-05}
        >>> anime_bangumi_char_score('bangumi_char/vision/4.jpg')
        {'vision': 0.9998877048492432, 'imagery': 8.732793503440917e-05, 'halfbody': 1.4264976925915107e-05, 'face': 1.0623419257171918e-05}
        >>> anime_bangumi_char_score('bangumi_char/imagery/5.jpg')
        {'vision': 0.07076334953308105, 'imagery': 0.9290977716445923, 'halfbody': 0.0001044218079186976, 'face': 3.4467317163944244e-05}
        >>> anime_bangumi_char_score('bangumi_char/imagery/6.jpg')
        {'vision': 2.2568268832401372e-05, 'imagery': 0.9999498128890991, 'halfbody': 2.1810528778587468e-05, 'face': 5.879474429093534e-06}
        >>> anime_bangumi_char_score('bangumi_char/imagery/7.jpg')
        {'vision': 3.260669109295122e-05, 'imagery': 0.9999510049819946, 'halfbody': 1.2321036592766177e-05, 'face': 4.025227553938748e-06}
        >>> anime_bangumi_char_score('bangumi_char/imagery/8.jpg')
        {'vision': 1.4251427273848094e-05, 'imagery': 0.999957799911499, 'halfbody': 2.4273678718600422e-05, 'face': 3.6884023302263813e-06}
        >>> anime_bangumi_char_score('bangumi_char/halfbody/9.jpg')
        {'vision': 3.880981603288092e-05, 'imagery': 0.0002326338435523212, 'halfbody': 0.9996368885040283, 'face': 9.164971561403945e-05}
        >>> anime_bangumi_char_score('bangumi_char/halfbody/10.jpg')
        {'vision': 0.00020793956355191767, 'imagery': 0.13438372313976288, 'halfbody': 0.8652494549751282, 'face': 0.000158855298650451}
        >>> anime_bangumi_char_score('bangumi_char/halfbody/11.jpg')
        {'vision': 0.000238816806813702, 'imagery': 0.3589179217815399, 'halfbody': 0.6406960487365723, 'face': 0.0001471740542910993}
        >>> anime_bangumi_char_score('bangumi_char/halfbody/12.jpg')
        {'vision': 0.002255884697660804, 'imagery': 0.08208147436380386, 'halfbody': 0.9152728915214539, 'face': 0.00038967153523117304}
        >>> anime_bangumi_char_score('bangumi_char/face/13.jpg')
        {'vision': 9.227699592884164e-06, 'imagery': 1.0835404282261152e-05, 'halfbody': 5.1437502406770363e-05, 'face': 0.9999284744262695}
        >>> anime_bangumi_char_score('bangumi_char/face/14.jpg')
        {'vision': 1.2125529792683665e-05, 'imagery': 1.0218892384727951e-05, 'halfbody': 0.00011914174683624879, 'face': 0.9998584985733032}
        >>> anime_bangumi_char_score('bangumi_char/face/15.jpg')
        {'vision': 1.2007669283775613e-05, 'imagery': 1.6357082131435163e-05, 'halfbody': 5.3068713896209374e-05, 'face': 0.9999185800552368}
        >>> anime_bangumi_char_score('bangumi_char/face/16.jpg')
        {'vision': 1.066640925273532e-05, 'imagery': 9.529400813335087e-06, 'halfbody': 4.089402500540018e-05, 'face': 0.9999388456344604}
    """
    return classify_predict_score(image, _REPO_ID, model_name)


def anime_bangumi_char(image: ImageTyping, model_name: str = _DEFAULT_MODEL_NAME) -> Tuple[str, float]:
    """
    Get the primary anime bangumi character type and its score.

    :param image: The input image.
    :type image: ImageTyping
    :param model_name: The model name. Default is 'mobilenetv3_v0_dist'.
    :type model_name: str
    :return: A tuple with the primary type and its score.
    :rtype: Tuple[str, float]

    Examples::
        >>> from imgutils.validate import anime_bangumi_char
        >>>
        >>> anime_bangumi_char('bangumi_char/vision/1.jpg')
        ('vision', 0.9998525381088257)
        >>> anime_bangumi_char('bangumi_char/vision/2.jpg')
        ('vision', 0.9997243285179138)
        >>> anime_bangumi_char('bangumi_char/vision/3.jpg')
        ('vision', 0.9998849630355835)
        >>> anime_bangumi_char('bangumi_char/vision/4.jpg')
        ('vision', 0.9998877048492432)
        >>> anime_bangumi_char('bangumi_char/imagery/5.jpg')
        ('imagery', 0.9290977716445923)
        >>> anime_bangumi_char('bangumi_char/imagery/6.jpg')
        ('imagery', 0.9999498128890991)
        >>> anime_bangumi_char('bangumi_char/imagery/7.jpg')
        ('imagery', 0.9999510049819946)
        >>> anime_bangumi_char('bangumi_char/imagery/8.jpg')
        ('imagery', 0.999957799911499)
        >>> anime_bangumi_char('bangumi_char/halfbody/9.jpg')
        ('halfbody', 0.9996368885040283)
        >>> anime_bangumi_char('bangumi_char/halfbody/10.jpg')
        ('halfbody', 0.8652494549751282)
        >>> anime_bangumi_char('bangumi_char/halfbody/11.jpg')
        ('halfbody', 0.6406959295272827)
        >>> anime_bangumi_char('bangumi_char/halfbody/12.jpg')
        ('halfbody', 0.9152728915214539)
        >>> anime_bangumi_char('bangumi_char/face/13.jpg')
        ('face', 0.9999284744262695)
        >>> anime_bangumi_char('bangumi_char/face/14.jpg')
        ('face', 0.9998584985733032)
        >>> anime_bangumi_char('bangumi_char/face/15.jpg')
        ('face', 0.9999185800552368)
        >>> anime_bangumi_char('bangumi_char/face/16.jpg')
        ('face', 0.9999388456344604)
    """
    return classify_predict(image, _REPO_ID, model_name)
