from typing import Optional

import numpy as np
from PIL import ImageColor, Image

from .image import ImageTyping
from .image import load_image
from .layer import istack

__all__ = [
    'grid_background',
    'grid_transparent',
]


def grid_background(height, width, step: Optional[int] = None,
                    forecolor: str = 'lightgrey', backcolor: str = 'white'):
    """
    Overview:
        Create an image with black and white squares, which can be used to complement
        transparent areas of an image with a transparent background.

    :param height: Height of image.
    :param width: Width of image.
    :param step: The step size of the grid in pixels. The default value is ``None``,
        which means that this function will automatically generate a suitable step value.
    :param forecolor: Color of the fore grids.
    :param backcolor: Color of the back grids.
    :return: A RGBA image which contains the grids.
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    if step is None:
        step = int((height * width / 800) ** 0.5)

    for x in range(0, height, step):
        for y in range(0, width, step):
            if (x // step + y // step) % 2 == 0:  # back
                img[x: x + step, y:y + step, :] = ImageColor.getrgb(backcolor)
            else:  # fore
                img[x: x + step, y:y + step, :] = ImageColor.getrgb(forecolor)

    return Image.fromarray(img, mode='RGB').convert('RGBA')


def grid_transparent(image: ImageTyping, step: Optional[int] = None,
                     forecolor: str = 'lightgrey', backcolor: str = 'white'):
    """
    Overview:
        Add a gridded background to an image with a transparent background.

    :param image: Original image.
    :param step: The step size of the grid in pixels. The default value is ``None``,
        which means that this function will automatically generate a suitable step value.
    :param forecolor: Color of the fore grids.
    :param backcolor: Color of the back grids.
    :return: A RGB image which contains the grids and the original image.

    .. note::
        In this document, :func:`grid_transparent` is the default option used to
        accurately present the state of the generated image, as shown in the following figure

        .. image:: grid_transparent.plot.py.svg
           :align: center

    """
    image = load_image(image, force_background=None)
    retval = grid_background(image.height, image.width, step, forecolor, backcolor)
    return istack(retval, image).convert('RGB')
