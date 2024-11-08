"""
Overview:
    Anime character segmentation, based on https://huggingface.co/skytnt/anime-seg .
"""

import cv2
import huggingface_hub
import numpy as np

from ..data import ImageTyping, load_image, istack
from ..utils import ts_lru_cache
from ..utils.onnxruntime import open_onnx_model


@ts_lru_cache()
def _get_model():
    return open_onnx_model(huggingface_hub.hf_hub_download("skytnt/anime-seg", "isnetis.onnx"))


def get_isnetis_mask(image: ImageTyping, scale: int = 1024):
    """
    Overview:
        Get mask with isnetis.

    :param image: Original image (assume its size is ``(H, W)``).
    :param scale: Scale when passing it into neural network. Default is ``1024``,
        inspired by https://huggingface.co/spaces/skytnt/anime-remove-background/blob/main/app.py#L8 .
    :return: Get a mask with all the pixels, which shape is ``(H, W)``.
    """
    image = np.asarray(load_image(image, mode='RGB'))
    image = (image / 255).astype(np.float32)
    h, w = h0, w0 = image.shape[:-1]
    h, w = (scale, int(scale * w / h)) if h > w else (int(scale * h / w), scale)
    ph, pw = scale - h, scale - w
    img_input = np.zeros([scale, scale, 3], dtype=np.float32)
    img_input[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w] = cv2.resize(image, (w, h))
    img_input = np.transpose(img_input, (2, 0, 1))
    img_input = img_input[np.newaxis, :]
    mask = _get_model().run(None, {'img': img_input})[0][0]
    mask = np.transpose(mask, (1, 2, 0))
    mask = mask[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w]
    mask = cv2.resize(mask, (w0, h0))[:, :, np.newaxis]
    return mask.reshape(*mask.shape[:-1])


def segment_with_isnetis(image: ImageTyping, background: str = 'lime', scale: int = 1024):
    """
    Overview:
        Segment image with pure color background.

    :param image: Original image (assume its size is ``(H, W)``).
    :param background: Background color for padding. Default is ``lime`` which represents ``#00ff00``.
    :param scale: Scale when passing it into neural network. Default is ``1024``,
        inspired by https://huggingface.co/spaces/skytnt/anime-remove-background/blob/main/app.py#L8 .
    :return: The mask and An RGB image featuring a pure-colored background along with a segmented image.

    Examples::
        >>> from imgutils.segment import segment_with_isnetis
        >>>
        >>> mask_, image_ = segment_with_isnetis('hutao.png')
        >>> image_.save('hutao_seg.png')
        >>>
        >>> mask_, image_ = segment_with_isnetis('skadi.jpg', background='white')  # white background
        >>> image_.save('skadi_seg.jpg')

        The result should be

        .. image:: isnetis_color.plot.py.svg
           :align: center

    """
    image = load_image(image, mode='RGB')
    mask = get_isnetis_mask(image, scale)
    return mask, istack((background, 1.0), (image, mask)).convert('RGB')


def segment_rgba_with_isnetis(image: ImageTyping, scale: int = 1024):
    """
    Overview:
        Segment image with transparent background.

    :param image: Original image (assume its size is ``(H, W)``).
    :param scale: Scale when passing it into neural network. Default is ``1024``,
        inspired by https://huggingface.co/spaces/skytnt/anime-remove-background/blob/main/app.py#L8 .
    :return: The mask and An RGBA image featuring a transparent background along with a segmented image.

    Examples::
        >>> from imgutils.segment import segment_rgba_with_isnetis
        >>>
        >>> mask_, image_ = segment_rgba_with_isnetis('hutao.png')
        >>> image_.save('hutao_seg.png')
        >>>
        >>> mask_, image_ = segment_rgba_with_isnetis('skadi.jpg')
        >>> image_.save('skadi_seg.png')

        The result should be

        .. image:: isnetis_trans.plot.py.svg
           :align: center

    """
    image = load_image(image, mode='RGB')
    mask = get_isnetis_mask(image, scale)
    return mask, istack((image, mask))
