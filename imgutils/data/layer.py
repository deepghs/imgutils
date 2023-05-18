from typing import Union, Tuple, Optional

import numpy as np
from PIL import ImageColor, Image

from .image import load_image, ImageTyping

__all__ = [
    'istack',
]


def _load_image_or_color(image) -> Union[str, Image.Image]:
    if isinstance(image, str):
        try:
            _ = ImageColor.getrgb(image)
        except ValueError:
            pass
        else:
            return image

    return load_image(image, mode='RGBA', force_background=None)


def _process(item):
    if isinstance(item, tuple):
        image, alpha = item
    else:
        image, alpha = item, 1

    return _load_image_or_color(image), alpha


_AlphaTyping = Union[float, np.ndarray]


def _add_alpha(image: Image.Image, alpha: _AlphaTyping) -> Image.Image:
    data = np.array(image.convert('RGBA')).astype(np.float32)
    data[:, :, 3] = (data[:, :, 3] * alpha).clip(0, 255)
    return Image.fromarray(data.astype(np.uint8), mode='RGBA')


def istack(*items: Union[ImageTyping, str, Tuple[ImageTyping, _AlphaTyping], Tuple[str, _AlphaTyping]],
           size: Optional[Tuple[int, int]] = None) -> Image.Image:
    """
    Overview:
        Layer multiple images (which may contain transparent areas) and
        color blocks together into a new image, similar to a layering technique in PS.

    :param items: The layers that need to be stacked. If a PIL object or the file path of an image is given,
        the image will be used as a layer; if a color is given, the color will be used as a layer.
        Additionally, if a tuple is given, the second element represents the transparency,
        with a value range of :math:`\\left[0, 1\\right]`. It can be a float type or a two-dimensional numpy array
        in the format of ``float32[H, W]`` which represents the transparency of each position.
    :param size: The size of the target image. By default, the size of the first image object in the `items` list
        will be used. However, when all layers are solid colors, this parameter is required.
    :return: Stacked image.

    Examples::
        >>> from imgutils.data import istack
        >>>
        >>> # pure color
        >>> istack('lime', 'nian.png').save('nian_lime.png')
        >>>
        >>> # transparency
        >>> istack(('yellow', 0.5), ('nian.png', 0.9)).save('nian_trans.png')
        >>>
        >>> # custom mask
        >>> import numpy as np
        >>> from PIL import Image
        >>> width, height = Image.open('nian.png').size
        >>> hs1 = (1 - np.abs(np.linspace(-1 / 3, 1, height))) ** 0.5
        >>> ws1 = (1 - np.abs(np.linspace(-1, 1, width))) ** 0.5
        >>> nian_mask = hs1[..., None] * ws1  # HxW
        >>> istack(('nian.png', nian_mask)).save('nian_mask.png')

        The result should be

        .. image:: grid_istack.plot.py.svg
           :align: center
    """
    if size is None:
        height, width = None, None
        items = list(map(_process, items))
        for item, alpha in items:
            if isinstance(item, Image.Image):
                height, width = item.height, item.width
                break
    else:
        width, height = size

    if height is None:
        raise ValueError('Unable to determine image size, please make sure '
                         'you have provided at least one image object (image path or PIL object).')

    retval = Image.fromarray(np.zeros((height, width, 4), dtype=np.uint8), mode='RGBA')
    for item, alpha in items:
        if isinstance(item, str):
            current = Image.new("RGBA", (width, height), item)
        elif isinstance(item, Image.Image):
            current = item
        else:
            assert False, f'Invalid type - {item!r}. If you encounter this situation, ' \
                          f'it means there is a bug in the code. Please contact the developer.'  # pragma: no cover

        current = _add_alpha(current, alpha)
        retval.paste(current, mask=current)

    return retval
