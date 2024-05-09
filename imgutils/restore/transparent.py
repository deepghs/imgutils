from typing import Optional

import numpy as np
from PIL import Image

from ..data.image import ImageTyping, load_image


def _has_alpha_channel(image: Image.Image) -> bool:
    return any(band in {'A', 'a', 'P'} for band in image.getbands())


def _rgba_preprocess(image: ImageTyping):
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
    assert pimage.mode == 'RGB'
    if alpha_mask is None:
        return pimage
    else:
        alpha_channel = (alpha_mask * 255.0).astype(np.uint8)[..., np.newaxis]
        channels = np.array(pimage)
        rgba_channels = np.concatenate([channels, alpha_channel], axis=-1)
        assert rgba_channels.shape == (*channels.shape[:-1], 4)
        return Image.fromarray(rgba_channels, mode='RGBA')
