"""
Overview:
    A model for classifying anime images into 4 classes (``3d``, ``bangumi``, ``comic`` and ``illustration``).

    The following are sample images for testing.

    .. image:: classify.plot.py.svg
        :align: center

    This is an overall benchmark of all the classification validation models:

    .. image:: classify_benchmark.plot.py.svg
        :align: center

    The models are hosted on
    `huggingface - deepghs/anime_classification <https://huggingface.co/deepghs/anime_classification>`_.
"""
from functools import lru_cache
from typing import Tuple, Optional, Dict

import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

from imgutils.data import rgb_encode, ImageTyping, load_image
from imgutils.utils import open_onnx_model

__all__ = [
    'anime_classify_score',
    'anime_classify',
]

_LABELS = ['3d', 'bangumi', 'comic', 'illustration']
_MODEL_NAMES = [
    'caformer_s36',
    'caformer_s36_plus',
    'mobilenetv3',
    'mobilenetv3_dist',
    'mobilenetv3_sce',
    'mobilenetv3_sce_dist',
    'mobilevitv2_150',
]
_DEFAULT_MODEL_NAME = 'mobilenetv3_sce_dist'


@lru_cache()
def _open_anime_classify_model(model_name):
    return open_onnx_model(hf_hub_download(
        f'deepghs/anime_classification',
        f'{model_name}/model.onnx',
    ))


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


def _raw_anime_classify(image: ImageTyping, model_name: str = _DEFAULT_MODEL_NAME):
    image = load_image(image, force_background='white', mode='RGB')
    input_ = _img_encode(image)[None, ...]
    output, = _open_anime_classify_model(model_name).run(['output'], {'input': input_})

    return output


def anime_classify_score(image: ImageTyping, model_name: str = _DEFAULT_MODEL_NAME) -> Dict[str, float]:
    """
    Overview:
        Predict the class of the given image, return the score with as a dict object.

    :param image: Image to classify.
    :param model_name: Model to use. Default is ``mobilenetv3_sce_dist``. All available models are listed
        on the benchmark plot above. If you need better accuracy, just set this to ``caformer_s36_plus``.
    :return: A dict with classes and scores.

    Examples::
        >>> from imgutils.validate import anime_classify_score
        >>>
        >>> anime_classify_score('classify/3d/1.jpg')
        {'3d': 0.9999048709869385, 'bangumi': 1.2998967577004805e-05, 'comic': 6.3774782574910205e-06, 'illustration': 7.573143375338987e-05}
        >>> anime_classify_score('classify/3d/2.jpg')
        {'3d': 1.0, 'bangumi': 2.3347255118794097e-12, 'comic': 5.393629119720966e-12, 'illustration': 6.71077689945454e-12}
        >>> anime_classify_score('classify/3d/3.jpg')
        {'3d': 0.9999587535858154, 'bangumi': 1.6608031728537753e-05, 'comic': 1.4294577340479009e-05, 'illustration': 1.0324462891730946e-05}
        >>> anime_classify_score('classify/bangumi/4.jpg')
        {'3d': 8.245967464404202e-09, 'bangumi': 0.9999991655349731, 'comic': 2.004386701059957e-08, 'illustration': 8.202430876735889e-07}
        >>> anime_classify_score('classify/bangumi/5.jpg')
        {'3d': 6.440834113163874e-05, 'bangumi': 0.9982288479804993, 'comic': 1.4121969797997735e-05, 'illustration': 0.001692703110165894}
        >>> anime_classify_score('classify/bangumi/6.jpg')
        {'3d': 2.3443080159404883e-14, 'bangumi': 1.0, 'comic': 5.647845608075866e-14, 'illustration': 6.008537851293072e-13}
        >>> anime_classify_score('classify/comic/7.jpg')
        {'3d': 4.029740221408245e-18, 'bangumi': 4.658470278842451e-18, 'comic': 1.0, 'illustration': 2.0487814569869478e-11}
        >>> anime_classify_score('classify/comic/8.jpg')
        {'3d': 1.019530813939351e-11, 'bangumi': 1.5961519215720865e-12, 'comic': 1.0, 'illustration': 2.2395576712574972e-11}
        >>> anime_classify_score('classify/comic/9.jpg')
        {'3d': 2.1237236958165234e-13, 'bangumi': 2.3246717593440602e-14, 'comic': 1.0, 'illustration': 3.4230233231236085e-11}
        >>> anime_classify_score('classify/illustration/10.jpg')
        {'3d': 0.00026091927429661155, 'bangumi': 0.00011691388499457389, 'comic': 8.51359436637722e-05, 'illustration': 0.9995369911193848}
        >>> anime_classify_score('classify/illustration/11.jpg')
        {'3d': 6.014750475458186e-09, 'bangumi': 2.3536564697224094e-07, 'comic': 7.933858796604909e-06, 'illustration': 0.999991774559021}
        >>> anime_classify_score('classify/illustration/12.jpg')
        {'3d': 3.153582292725332e-05, 'bangumi': 0.0001071861624950543, 'comic': 5.665345452143811e-05, 'illustration': 0.999804675579071}
    """
    output = _raw_anime_classify(image, model_name)
    values = dict(zip(_LABELS, map(lambda x: x.item(), output[0])))
    return values


def anime_classify(image: ImageTyping, model_name: str = _DEFAULT_MODEL_NAME) -> Tuple[str, float]:
    """
    Overview:
        Predict the class of the given image, return the class and its score.

    :param image: Image to classify.
    :param model_name: Model to use. Default is ``mobilenetv3_sce_dist``. All available models are listed
        on the benchmark plot above. If you need better accuracy, just set this to ``caformer_s36_plus``.
    :return: A tuple contains the class and its score.

    Examples::
        >>> from imgutils.validate import anime_classify
        >>>
        >>> anime_classify('classify/3d/1.jpg')
        ('3d', 0.9999048709869385)
        >>> anime_classify('classify/3d/2.jpg')
        ('3d', 1.0)
        >>> anime_classify('classify/3d/3.jpg')
        ('3d', 0.9999587535858154)
        >>> anime_classify('classify/bangumi/4.jpg')
        ('bangumi', 0.9999991655349731)
        >>> anime_classify('classify/bangumi/5.jpg')
        ('bangumi', 0.9982288479804993)
        >>> anime_classify('classify/bangumi/6.jpg')
        ('bangumi', 1.0)
        >>> anime_classify('classify/comic/7.jpg')
        ('comic', 1.0)
        >>> anime_classify('classify/comic/8.jpg')
        ('comic', 1.0)
        >>> anime_classify('classify/comic/9.jpg')
        ('comic', 1.0)
        >>> anime_classify('classify/illustration/10.jpg')
        ('illustration', 0.9995369911193848)
        >>> anime_classify('classify/illustration/11.jpg')
        ('illustration', 0.999991774559021)
        >>> anime_classify('classify/illustration/12.jpg')
        ('illustration', 0.999804675579071)
    """
    output = _raw_anime_classify(image, model_name)[0]
    max_id = np.argmax(output)
    return _LABELS[max_id], output[max_id].item()
