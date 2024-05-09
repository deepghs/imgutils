from typing import Optional

import numpy as np
import scipy.ndimage
from PIL import Image

from ..data.image import ImageTyping, load_image


def _has_alpha_channel(image: Image.Image) -> bool:
    """
    Check if the image has an alpha channel.

    :param image: The image to check.
    :type image: Image.Image

    :return: True if the image has an alpha channel, False otherwise.
    :rtype: bool
    """
    return any(band in {'A', 'a', 'P'} for band in image.getbands())


def _rgba_preprocess(image: ImageTyping):
    """
    Preprocess the image for RGBA conversion.

    :param image: The image to preprocess.
    :type image: ImageTyping

    :return: Preprocessed image and alpha mask.
    :rtype: Tuple[Image.Image, Optional[np.ndarray]]
    """
    image = load_image(image, force_background=None, mode=None)
    if _has_alpha_channel(image):
        image = image.convert('RGBA')
        pimage = image.convert('RGB')
        alpha_mask = np.array(image)[:, :, 3].astype(np.float32) / 255.0
    else:
        pimage = image.convert('RGB')
        alpha_mask = None

    return pimage, alpha_mask


def _rgba_postprocess(pimage, alpha_mask: Optional[np.ndarray] = None):
    """
    Postprocess the image after RGBA conversion.

    :param pimage: The processed image.
    :type pimage: Image.Image

    :param alpha_mask: The alpha mask.
    :type alpha_mask: Optional[np.ndarray]

    :return: Postprocessed image.
    :rtype: Image.Image
    """
    assert pimage.mode == 'RGB'
    if alpha_mask is None:
        return pimage
    else:
        channels = np.array(pimage)
        alpha_mask = scipy.ndimage.zoom(
            alpha_mask,
            np.array(channels.shape[:2]) / np.array(alpha_mask.shape),
            mode='nearest',
            order=1,
        )
        alpha_channel = (alpha_mask * 255.0).astype(np.uint8)[..., np.newaxis]
        rgba_channels = np.concatenate([channels, alpha_channel], axis=-1)
        assert rgba_channels.shape == (*channels.shape[:-1], 4)
        return Image.fromarray(rgba_channels, mode='RGBA')
