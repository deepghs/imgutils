"""
Overview:
    Tool for determining the NSFW (Not Safe for Work) type of a given image, which includes
    five categories: ``drawings``, ``hentai``, ``neutral``, ``porn``, and ``sexy``.
    It is based on `infinitered/nsfwjs <https://github.com/infinitered/nsfwjs>`_,
    a high-performance model originally in tfjs format, which has been converted to onnx format
    for deployment, making it suitable for mobile applications.

    .. collapse:: The following are sample images for testing. (WARNING: NSFW!!!)

        .. image:: nsfw.plot.py.svg
            :align: center

    This is an overall benchmark of all the  validation models:

    .. image:: nsfw_benchmark.plot.py.svg
        :align: center
"""
from typing import Mapping, Tuple

import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

from ..data import load_image, ImageTyping
from ..utils import open_onnx_model, ts_lru_cache

__all__ = [
    'nsfw_pred_score',
    'nsfw_pred',
]

_MODELS = [
    ('nsfwjs', 224),
    ('inception_v3', 299),
]
_MODEL_NAMES = [name for name, _ in _MODELS]
_DEFAULT_MODEL_NAME = 'nsfwjs'
_MODEL_TO_SIZE = dict(_MODELS)


@ts_lru_cache()
def _open_nsfw_model(model: str = _DEFAULT_MODEL_NAME):
    """
    Opens the NSFW model for performing NSFW predictions.

    The function downloads the ONNX model file from the Hugging Face Hub using the specified `model` name.
    The model is then loaded and returned.

    :param model: The name of the NSFW model to open. (default: ``nsfwjs``)
    :type model: str

    :return: The loaded NSFW model.
    :rtype: onnxruntime.InferenceSession
    """
    return open_onnx_model(hf_hub_download(
        'deepghs/imgutils-models',
        f'nsfw/{model}.onnx'
    ))


def _image_preprocess(image, size: int = 224) -> np.ndarray:
    """
    Preprocesses the image for NSFW prediction.

    The function loads the image, resizes it to the specified ``size``, and converts it to a numpy array.
    The pixel values are normalized to the range :math:`\left[0, 1\right]`.

    :param image: The image to preprocess.
    :type image: ImageTyping

    :param size: The size to resize the image to. (default: ``224``)
    :type size: int

    :return: The preprocessed image as a numpy array.
    :rtype: np.ndarray
    """
    image = load_image(image, mode='RGB').resize((size, size), Image.NEAREST)
    return (np.array(image) / 255.0)[None, ...]


_LABELS = ['drawings', 'hentai', 'neutral', 'porn', 'sexy']


def _raw_scores(image: ImageTyping, model_name: str = _DEFAULT_MODEL_NAME) -> np.ndarray:
    """
    Computes the raw prediction scores for the NSFW categories.

    The function preprocesses the image, passes it through the specified NSFW model, and extracts the output scores.
    The scores represent the predicted probability for each NSFW category.

    :param image: The image to compute scores for.
    :type image: ImageTyping

    :param model_name: The name of the NSFW model to use. (default: ``nsfwjs``)
    :type model_name: str

    :return: The raw prediction scores as a numpy array.
    :rtype: np.ndarray
    """
    input_ = _image_preprocess(image, _MODEL_TO_SIZE[model_name]).astype(np.float32)
    output_, = _open_nsfw_model(model_name).run(['dense_3'], {'input_1': input_})
    return output_[0]


def nsfw_pred_score(image: ImageTyping, model_name: str = _DEFAULT_MODEL_NAME) -> Mapping[str, float]:
    """
    Computes the NSFW prediction scores for the input image.

    The function returns a mapping of NSFW category labels to their corresponding prediction scores.
    The scores represent the predicted probability for each NSFW category.

    :param image: The image to compute prediction scores for.
    :type image: ImageTyping

    :param model_name: The name of the NSFW model to use. (default: ``nsfwjs``)
    :type model_name: str

    :return: The NSFW prediction scores as a mapping of labels to scores.
    :rtype: Mapping[str, float]

    Examples::
        >>> from imgutils.validate import nsfw_pred_score
        >>>
        >>> nsfw_pred_score('nsfw/drawings/1.jpg')
        {'drawings': 0.9970946311950684, 'hentai': 0.00198739324696362, 'neutral': 0.000894528697244823, 'porn': 1.4315058251668233e-05, 'sexy': 9.099447197513655e-06}
        >>> nsfw_pred_score('nsfw/drawings/2.jpg')
        {'drawings': 0.9282580614089966, 'hentai': 0.061733175069093704, 'neutral': 0.008979619480669498, 'porn': 0.0007789491210132837, 'sexy': 0.0002501663693692535}
        >>> nsfw_pred_score('nsfw/drawings/3.jpg')
        {'drawings': 0.7945129871368408, 'hentai': 0.2044062316417694, 'neutral': 0.0005603990866802633, 'porn': 0.0004847997915931046, 'sexy': 3.564094367902726e-05}
        >>> nsfw_pred_score('nsfw/drawings/4.jpg')
        {'drawings': 0.7977773547172546, 'hentai': 0.01352313905954361, 'neutral': 0.18791256844997406, 'porn': 0.0004888656549155712, 'sexy': 0.00029804420773871243}
        >>> nsfw_pred_score('nsfw/hentai/5.jpg')
        {'drawings': 0.04498734697699547, 'hentai': 0.9509441256523132, 'neutral': 2.4087972633424215e-05, 'porn': 0.003999904729425907, 'sexy': 4.4542059185914695e-05}
        >>> nsfw_pred_score('nsfw/hentai/6.jpg')
        {'drawings': 0.002892113756388426, 'hentai': 0.982390284538269, 'neutral': 6.02520776737947e-06, 'porn': 0.014633022248744965, 'sexy': 7.858086610212922e-05}
        >>> nsfw_pred_score('nsfw/hentai/7.jpg')
        {'drawings': 0.002532319398596883, 'hentai': 0.9887337684631348, 'neutral': 6.231979568838142e-06, 'porn': 0.008699454367160797, 'sexy': 2.8187158022774383e-05}
        >>> nsfw_pred_score('nsfw/hentai/8.jpg')
        {'drawings': 0.03564726561307907, 'hentai': 0.954788088798523, 'neutral': 7.343036850215867e-05, 'porn': 0.009289607405662537, 'sexy': 0.00020158555707894266}
        >>> nsfw_pred_score('nsfw/neutral/9.jpg')
        {'drawings': 0.006372362840920687, 'hentai': 0.006019102409482002, 'neutral': 0.9694945812225342, 'porn': 0.015214097686111927, 'sexy': 0.002899901708588004}
        >>> nsfw_pred_score('nsfw/neutral/10.jpg')
        {'drawings': 0.0004039364866912365, 'hentai': 0.00012730166781693697, 'neutral': 0.987038791179657, 'porn': 0.007135333959013224, 'sexy': 0.005294707603752613}
        >>> nsfw_pred_score('nsfw/neutral/11.jpg')
        {'drawings': 0.06964848190546036, 'hentai': 0.0014777459437027574, 'neutral': 0.9276643395423889, 'porn': 0.0003031621454283595, 'sexy': 0.0009063396137207747}
        >>> nsfw_pred_score('nsfw/neutral/12.jpg')
        {'drawings': 0.00028707628371194005, 'hentai': 0.00010888021643040702, 'neutral': 0.9992460012435913, 'porn': 0.00015473493840545416, 'sexy': 0.0002033217460848391}
        >>> nsfw_pred_score('nsfw/porn/13.jpg')
        {'drawings': 4.563037691696081e-06, 'hentai': 0.008058490231633186, 'neutral': 0.00044566826545633376, 'porn': 0.937960684299469, 'sexy': 0.05353058874607086}
        >>> nsfw_pred_score('nsfw/porn/14.jpg')
        {'drawings': 3.364063445587817e-07, 'hentai': 0.00562260951846838, 'neutral': 0.00012077406427124515, 'porn': 0.9897090792655945, 'sexy': 0.004547217860817909}
        >>> nsfw_pred_score('nsfw/porn/15.jpg')
        {'drawings': 8.564737981942017e-06, 'hentai': 0.016690678894519806, 'neutral': 0.001258736359886825, 'porn': 0.9766013622283936, 'sexy': 0.005440687295049429}
        >>> nsfw_pred_score('nsfw/porn/16.jpg')
        {'drawings': 1.4481674952548929e-05, 'hentai': 0.01861923187971115, 'neutral': 0.0008914825739338994, 'porn': 0.9674761295318604, 'sexy': 0.012998746708035469}
        >>> nsfw_pred_score('nsfw/sexy/17.jpg')
        {'drawings': 6.691116141155362e-05, 'hentai': 0.0007601747056469321, 'neutral': 0.0005019629606977105, 'porn': 0.039504989981651306, 'sexy': 0.9591660499572754}
        >>> nsfw_pred_score('nsfw/sexy/18.jpg')
        {'drawings': 0.0001652583305258304, 'hentai': 0.0002614929690025747, 'neutral': 0.020374108105897903, 'porn': 0.029394468292593956, 'sexy': 0.9498046040534973}
        >>> nsfw_pred_score('nsfw/sexy/19.jpg')
        {'drawings': 0.00016299057460855693, 'hentai': 0.004782819654792547, 'neutral': 0.002861740067601204, 'porn': 0.12280157208442688, 'sexy': 0.8693908452987671}
        >>> nsfw_pred_score('nsfw/sexy/20.jpg')
        {'drawings': 0.0001731760276015848, 'hentai': 6.304211274255067e-05, 'neutral': 0.03286275267601013, 'porn': 0.010648751631379128, 'sexy': 0.9562522172927856}
    """
    # noinspection PyTypeChecker
    return dict(zip(_LABELS, _raw_scores(image, model_name).tolist()))


def nsfw_pred(image: ImageTyping, model_name: str = _DEFAULT_MODEL_NAME) -> Tuple[str, float]:
    """
    Performs NSFW prediction on the input image.

    The function returns the predicted NSFW category label and its corresponding prediction score.
    The label represents the category with the highest predicted probability.

    :param image: The image to perform NSFW prediction on.
    :type image: ImageTyping

    :param model_name: The name of the NSFW model to use. (default: ``nsfwjs``)
    :type model_name: str

    :return: The predicted NSFW category label and its prediction score.
    :rtype: Tuple[str, float]

    Examples::
        >>> from imgutils.validate import nsfw_pred
        >>>
        >>> nsfw_pred('nsfw/drawings/1.jpg')
        ('drawings', 0.9970946311950684)
        >>> nsfw_pred('nsfw/drawings/2.jpg')
        ('drawings', 0.9282580614089966)
        >>> nsfw_pred('nsfw/drawings/3.jpg')
        ('drawings', 0.7945129871368408)
        >>> nsfw_pred('nsfw/drawings/4.jpg')
        ('drawings', 0.7977773547172546)
        >>> nsfw_pred('nsfw/hentai/5.jpg')
        ('hentai', 0.9509441256523132)
        >>> nsfw_pred('nsfw/hentai/6.jpg')
        ('hentai', 0.982390284538269)
        >>> nsfw_pred('nsfw/hentai/7.jpg')
        ('hentai', 0.9887337684631348)
        >>> nsfw_pred('nsfw/hentai/8.jpg')
        ('hentai', 0.954788088798523)
        >>> nsfw_pred('nsfw/neutral/9.jpg')
        ('neutral', 0.9694945812225342)
        >>> nsfw_pred('nsfw/neutral/10.jpg')
        ('neutral', 0.987038791179657)
        >>> nsfw_pred('nsfw/neutral/11.jpg')
        ('neutral', 0.9276643395423889)
        >>> nsfw_pred('nsfw/neutral/12.jpg')
        ('neutral', 0.9992460012435913)
        >>> nsfw_pred('nsfw/porn/13.jpg')
        ('porn', 0.937960684299469)
        >>> nsfw_pred('nsfw/porn/14.jpg')
        ('porn', 0.9897090792655945)
        >>> nsfw_pred('nsfw/porn/15.jpg')
        ('porn', 0.9766013622283936)
        >>> nsfw_pred('nsfw/porn/16.jpg')
        ('porn', 0.9674761295318604)
        >>> nsfw_pred('nsfw/sexy/17.jpg')
        ('sexy', 0.9591660499572754)
        >>> nsfw_pred('nsfw/sexy/18.jpg')
        ('sexy', 0.9498046040534973)
        >>> nsfw_pred('nsfw/sexy/19.jpg')
        ('sexy', 0.8693908452987671)
        >>> nsfw_pred('nsfw/sexy/20.jpg')
        ('sexy', 0.9562522172927856)
    """
    scores = _raw_scores(image, model_name)
    maxid = np.argmax(scores)
    return _LABELS[maxid], scores[maxid].item()
