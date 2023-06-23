"""
Overview:
    A utility for squeezing a specified region of an image.
"""
from typing import Optional

import numpy as np
from scipy import ndimage

from ..data import ImageTyping, load_image


def squeeze(image: ImageTyping, mask: np.ndarray):
    """
    Extracts the corresponding region from the original image based on the provided mask (HxW format)
    and crops the image to fit the mask tightly.

    :param image: The input image.
    :type image: ImageTyping

    :param mask: The mask representing the region of interest. It should be a NumPy array with shape ``(H, W)``.
    :type mask: np.ndarray

    :raises ValueError: If the shape of the image and mask do not match.

    :return: The cropped image that fits the mask tightly.
    :rtype: Image.Image

    Examples::
        >>> from PIL import Image
        >>> from imgutils.operate import squeeze
        >>>
        >>> origin = Image.open('jerry_with_space.png')
        >>> mask = ...  # set your custom mask, format: bool[H, W]
        >>>
        >>> squeezed = squeeze(origin, mask)

        This is the result:

        .. image:: squeeze.plot.py.svg
            :align: center
    """
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
    """
    Automatically crops the image based on the transparency of each pixel using the :func:`squeeze` function.

    :param image: The input image.
    :type image: ImageTyping

    :param threshold: The threshold value for pixel transparency. Pixels with transparency above this threshold
        will be considered as part of the region of interest. Default is ``0.7``.
    :type threshold: float

    :param median_filter: The size of the median filter kernel to apply to the transparency mask.
        A larger value helps reduce noise in the mask. Set to None or 0 to disable median filtering. Default is ``5``.
    :type median_filter: Optional[int]

    :return: The cropped image based on the transparency of each pixel.
    :rtype: Image.Image

    Examples::
        >>> from PIL import Image
        >>> from imgutils.operate import squeeze_with_transparency
        >>>
        >>> origin = Image.open('jerry_with_space.png')
        >>>
        >>> squeezed = squeeze_with_transparency(origin)

        This is the result:

        .. image:: squeeze_with_transparency.plot.py.svg
            :align: center
    """
    return squeeze(image, _get_mask_of_transparency(image, threshold, median_filter))
