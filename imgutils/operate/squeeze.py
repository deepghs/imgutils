from typing import Optional

import numpy as np
from scipy import ndimage

from ..data import ImageTyping, load_image


def squeeze(image: ImageTyping, mask: np.ndarray):
    image = load_image(image, force_background=None)
    if (image.height, image.width) != mask.shape:
        raise ValueError(f'Image shape not matched, '
                         f'{mask.shape!r} in mask but {(image.height, image.width)!r} in image.')

    mask = mask.astype(bool).astype(int)
    x_idx, = np.where(mask.sum(axis=1) > 0)
    x_min, x_max = x_idx.min(), x_idx.max()
    y_idx, = np.where(mask.sum(axis=0) > 0)
    y_min, y_max = y_idx.min(), y_idx.max()

    return image.crop((y_min, x_min, y_max, x_max))


def _get_mask_of_transparency(image: ImageTyping, threshold: float = 0.7, median_filter: Optional[int] = 5):
    image = load_image(image, mode='RGBA', force_background=None)
    mask = ((np.array(image)[:, :, 3].astype(np.float32) / 255.0) > threshold).astype(int)
    if median_filter is not None:
        mask = ndimage.median_filter(mask, size=median_filter)

    return mask.astype(bool)


def squeeze_with_transparency(image: ImageTyping, threshold: float = 0.7, median_filter: Optional[int] = 5):
    return squeeze(image, _get_mask_of_transparency(image, threshold, median_filter))
