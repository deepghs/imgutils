"""
Overview:
    Check if the images are polluted or safe.

    This is an overall benchmark of all the safe check models:

    .. image:: safe_benchmark.plot.py.svg
        :align: center

    Inspired from `mf666/shit-checker <https://huggingface.co/spaces/mf666/shit-checker>`_.
"""
import math
import random
from typing import Mapping, Tuple

import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

from ..data import ImageTyping, load_image
from ..utils import open_onnx_model, ts_lru_cache

__all__ = [
    'safe_check_score',
    'safe_check',
]

DEFAULT_MODEL = 'mobilenet.xs.v2'


@ts_lru_cache()
def _open_model(model_name):
    """
    Open the ONNX model specified by the model name.

    :param model_name: The name of the model.
    :type model_name: str
    :return: The opened ONNX model.
    :rtype: onnx.ModelProto
    """
    return open_onnx_model(hf_hub_download(
        repo_id='mf666/shit-checker',
        filename=f'{model_name}.onnx'
    ))


_DEFAULT_ORDER = 'HWC'


def _get_hwc_map(order_):
    return tuple(_DEFAULT_ORDER.index(c) for c in order_.upper())


def _encode_channels(image, channels_order='CHW'):
    array = np.asarray(image.convert('RGB'))
    array = np.transpose(array, _get_hwc_map(channels_order))
    array = (array / 255.0).astype(np.float32)
    assert array.dtype == np.float32
    return array


def _img_encode(image, size=(384, 384), normalize=(0.5, 0.5)):
    image = image.resize(size, Image.BILINEAR)
    data = _encode_channels(image, channels_order='CHW')

    if normalize is not None:
        mean_, std_ = normalize
        mean = np.asarray([mean_]).reshape((-1, 1, 1))
        std = np.asarray([std_]).reshape((-1, 1, 1))
        data = (data - mean) / std

    return data.astype(np.float32)


def _raw_predict(images, model_name=DEFAULT_MODEL):
    items = []
    for image in images:
        items.append(_img_encode(image.convert('RGB')))
    input_ = np.stack(items)
    output, = _open_model(model_name).run(['output'], {'input': input_})
    return output.mean(axis=0)


_LABELS = ['polluted', 'safe']


def _pred(image, model_name=DEFAULT_MODEL, max_batch_size=8):
    area = image.width * image.height
    batch_size = int(max(min(math.ceil(area / (384 * 384)) + 1, max_batch_size), 1))
    blocks = []
    for _ in range(batch_size):
        x0 = random.randint(0, max(0, image.width - 384))
        y0 = random.randint(0, max(0, image.height - 384))
        x1 = min(x0 + 384, image.width)
        y1 = min(y0 + 384, image.height)
        blocks.append(image.crop((x0, y0, x1, y1)))

    scores = _raw_predict(blocks, model_name)
    return scores


def safe_check_score(image: ImageTyping, model_name: str = DEFAULT_MODEL, max_batch_size: int = 8) \
        -> Mapping[str, float]:
    """
    Check the safety score of an image.

    :param image: The image to check.
    :type image: ImageTyping
    :param model_name: The name of the safety model.
    :type model_name: str
    :param max_batch_size: The maximum batch size for prediction.
    :type max_batch_size: int
    :return: A mapping of safety labels and their corresponding scores.
    :rtype: Mapping[str, float]
    """
    image = load_image(image)
    _pred_result = _pred(image, model_name, max_batch_size)
    return dict(zip(['polluted', 'safe'], map(lambda x: x.item(), _pred_result)))


def safe_check(image: ImageTyping, model_name: str = DEFAULT_MODEL, max_batch_size: int = 8) \
        -> Tuple[str, float]:
    """
    Check the safety label and score of an image.

    :param image: The image to check.
    :type image: ImageTyping
    :param model_name: The name of the safety model.
    :type model_name: str
    :param max_batch_size: The maximum batch size for prediction.
    :type max_batch_size: int
    :return: A tuple containing the safety label and score.
    :rtype: Tuple[str, float]
    """
    image = load_image(image)
    _pred_result = _pred(image, model_name, max_batch_size)
    id_ = _pred_result.argmax().item()
    return _LABELS[id_], _pred_result[id_].item()
