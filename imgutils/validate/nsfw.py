"""
Overview:
    Tool for determining the NSFW (Not Safe for Work) type of a given image, which includes
    five categories: 'drawings', 'hentai', 'neutral', 'porn', and 'sexy'.
    It is based on `infinitered/nsfwjs <https://github.com/infinitered/nsfwjs>`_,
    a high-performance model originally in tfjs format, which has been converted to onnx format
    for deployment, making it suitable for mobile applications.

    The following are sample images for testing.

    .. image:: nsfw.plot.py.svg
        :align: center

    This is an overall benchmark of all the rating validation models:

    .. image:: nsfw_benchmark.plot.py.svg
        :align: center
"""
from functools import lru_cache
from typing import Mapping, Tuple

import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

from ..data import load_image, ImageTyping
from ..utils import open_onnx_model

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


@lru_cache()
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
    """
    scores = _raw_scores(image, model_name)
    maxid = np.argmax(scores)
    return _LABELS[maxid], scores[maxid].item()
