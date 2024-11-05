"""
Overview:
    A model for classifying anime images into 5 classes (``3d``, ``bangumi``, ``comic``, ``illustration``, and ``not_painting``).

    The following are sample images for testing.

    .. image:: classify.plot.py.svg
        :align: center

    This is an overall benchmark of all the classification validation models:

    .. image:: classify_benchmark.plot.py.svg
        :align: center

    The models are hosted on
    `huggingface - deepghs/anime_classification <https://huggingface.co/deepghs/anime_classification>`_.

    .. note::
        In older versions of models, there are 4 classes, which means ``not_painting`` do not exist.
"""
from typing import Tuple, Dict

from ..data import ImageTyping
from ..generic import classify_predict, classify_predict_score

__all__ = [
    'anime_classify_score',
    'anime_classify',
]

_DEFAULT_MODEL_NAME = 'mobilenetv3_v1.5_dist'
_REPO_ID = 'deepghs/anime_classification'


def anime_classify_score(image: ImageTyping, model_name: str = _DEFAULT_MODEL_NAME) -> Dict[str, float]:
    """
    Overview:
        Predict the class of the given image, return the score with as a dict object.

    :param image: Image to classify.
    :param model_name: Model to use. Default is ``mobilenetv3_v1.3_dist``. All available models are listed
        on the benchmark plot above. If you need better accuracy, just set this to ``caformer_s36_v1.3_focal``.
    :return: A dict with classes and scores.

    Examples::
        >>> from imgutils.validate import anime_classify_score
        >>>
        >>> anime_classify_score('classify/3d/1.jpg')
        {'3d': 0.8346158862113953, 'bangumi': 0.004201625939458609, 'comic': 0.0028638991061598063, 'illustration': 0.15633030235767365, 'not_painting': 0.001988308737054467}
        >>> anime_classify_score('classify/3d/2.jpg')
        {'3d': 0.9868855476379395, 'bangumi': 0.001178382197394967, 'comic': 0.00015886101755313575, 'illustration': 0.0005986307514831424, 'not_painting': 0.011178601533174515}
        >>> anime_classify_score('classify/3d/3.jpg')
        {'3d': 0.9933090209960938, 'bangumi': 0.0012440024875104427, 'comic': 0.00040085514774546027, 'illustration': 0.004924307577311993, 'not_painting': 0.00012189441622467712}
        >>> anime_classify_score('classify/bangumi/4.jpg')
        {'3d': 0.00031298911198973656, 'bangumi': 0.9968050718307495, 'comic': 5.182305903872475e-05, 'illustration': 0.0027923565357923508, 'not_painting': 3.7805559259140864e-05}
        >>> anime_classify_score('classify/bangumi/5.jpg')
        {'3d': 0.0004650334012694657, 'bangumi': 0.996709942817688, 'comic': 3.736721191671677e-05, 'illustration': 0.0027629584074020386, 'not_painting': 2.4619508621981367e-05}
        >>> anime_classify_score('classify/bangumi/6.jpg')
        {'3d': 0.0003803370927926153, 'bangumi': 0.998649537563324, 'comic': 5.190127922105603e-05, 'illustration': 0.0008622839814051986, 'not_painting': 5.595230686594732e-05}
        >>> anime_classify_score('classify/comic/7.jpg')
        {'3d': 0.0004573142796289176, 'bangumi': 0.00031435859273187816, 'comic': 0.8671838641166687, 'illustration': 0.13199880719184875, 'not_painting': 4.563074617180973e-05}
        >>> anime_classify_score('classify/comic/8.jpg')
        {'3d': 7.153919796110131e-06, 'bangumi': 8.290010737255216e-05, 'comic': 0.9727378487586975, 'illustration': 0.027150526642799377, 'not_painting': 2.162296004826203e-05}
        >>> anime_classify_score('classify/comic/9.jpg')
        {'3d': 2.4933258828241378e-05, 'bangumi': 0.0004275702522136271, 'comic': 0.995402455329895, 'illustration': 0.002233930164948106, 'not_painting': 0.001911122351884842}
        >>> anime_classify_score('classify/illustration/10.jpg')
        {'3d': 0.1603819727897644, 'bangumi': 0.0007561995880678296, 'comic': 0.00017044576816260815, 'illustration': 0.838487982749939, 'not_painting': 0.0002034590725088492}
        >>> anime_classify_score('classify/illustration/11.jpg')
        {'3d': 0.005001617129892111, 'bangumi': 0.000932251859921962, 'comic': 0.009352140128612518, 'illustration': 0.9846979379653931, 'not_painting': 1.6018555470509455e-05}
        >>> anime_classify_score('classify/illustration/12.jpg')
        {'3d': 0.004064667969942093, 'bangumi': 9.464051254326478e-05, 'comic': 0.025772539898753166, 'illustration': 0.9699516296386719, 'not_painting': 0.00011656546121230349}
        >>> anime_classify_score('classify/not_painting/13.jpg')
        {'3d': 5.287263775244355e-05, 'bangumi': 3.370255853951676e-06, 'comic': 0.01098843663930893, 'illustration': 0.0031668643932789564, 'not_painting': 0.9857884049415588}
        >>> anime_classify_score('classify/not_painting/14.jpg')
        {'3d': 7.499273488065228e-05, 'bangumi': 2.8419872251106426e-05, 'comic': 0.0003471920208539814, 'illustration': 0.029472889378666878, 'not_painting': 0.9700765609741211}
        >>> anime_classify_score('classify/not_painting/15.jpg')
        {'3d': 0.0012387704337015748, 'bangumi': 0.001172148622572422, 'comic': 9.787473391043022e-05, 'illustration': 0.003680602880194783, 'not_painting': 0.9938107132911682}
    """
    return classify_predict_score(image, _REPO_ID, model_name)


def anime_classify(image: ImageTyping, model_name: str = _DEFAULT_MODEL_NAME) -> Tuple[str, float]:
    """
    Overview:
        Predict the class of the given image, return the class and its score.

    :param image: Image to classify.
    :param model_name: Model to use. Default is ``mobilenetv3_v1.3_dist``. All available models are listed
        on the benchmark plot above. If you need better accuracy, just set this to ``caformer_s36_v1.3_focal``.
    :return: A tuple contains the class and its score.

    Examples::
        >>> from imgutils.validate import anime_classify
        >>>
        >>> anime_classify('classify/3d/1.jpg')
        ('3d', 0.8346157073974609)
        >>> anime_classify('classify/3d/2.jpg')
        ('3d', 0.9868855476379395)
        >>> anime_classify('classify/3d/3.jpg')
        ('3d', 0.9933090209960938)
        >>> anime_classify('classify/bangumi/4.jpg')
        ('bangumi', 0.9968050718307495)
        >>> anime_classify('classify/bangumi/5.jpg')
        ('bangumi', 0.996709942817688)
        >>> anime_classify('classify/bangumi/6.jpg')
        ('bangumi', 0.998649537563324)
        >>> anime_classify('classify/comic/7.jpg')
        ('comic', 0.8671836853027344)
        >>> anime_classify('classify/comic/8.jpg')
        ('comic', 0.9727378487586975)
        >>> anime_classify('classify/comic/9.jpg')
        ('comic', 0.995402455329895)
        >>> anime_classify('classify/illustration/10.jpg')
        ('illustration', 0.8384883403778076)
        >>> anime_classify('classify/illustration/11.jpg')
        ('illustration', 0.9846979975700378)
        >>> anime_classify('classify/illustration/12.jpg')
        ('illustration', 0.9699516296386719)
        >>> anime_classify('classify/not_painting/13.jpg')
        ('not_painting', 0.9857884049415588)
        >>> anime_classify('classify/not_painting/14.jpg')
        ('not_painting', 0.9700766801834106)
        >>> anime_classify('classify/not_painting/15.jpg')
        ('not_painting', 0.9938107132911682)
    """
    return classify_predict(image, _REPO_ID, model_name)
