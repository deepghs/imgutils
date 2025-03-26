"""
Image padding and resizing utilities.

This module provides functions for padding and resizing images to specified dimensions
while maintaining aspect ratio. It includes utilities for parsing size specifications,
color values, and handling different image modes.
"""

from typing import Union, Tuple, Literal

from PIL import ImageColor, Image

from .image import ImageTyping, load_image

__all__ = [
    'pad_image_to_size',
]


def _parse_size(size: Union[Tuple[int, int], int]):
    """
    Parse size parameter into a tuple of width and height.

    :param size: Size specification as an integer or tuple of two integers
    :type size: Union[Tuple[int, int], int]

    :return: Tuple containing width and height
    :rtype: Tuple[int, int]

    :raises TypeError: If size is not an int or tuple/list of two ints
    """
    if isinstance(size, int):
        return size, size
    elif isinstance(size, (list, tuple)) and len(size) == 2:
        return int(size[0]), int(size[1])
    else:
        raise TypeError("Size must be int or tuple of two ints")


def _parse_color_to_rgba(color: Union[str, int, Tuple[int, int, int], Tuple[int, int, int, int]]):
    """
    Convert various color formats to RGBA tuple.

    :param color: Color specification (string, integer, or tuple/list)
    :type color: Union[str, int, Tuple[int, int, int], Tuple[int, int, int, int]]

    :return: RGBA color tuple
    :rtype: Tuple[int, int, int, int]

    :raises TypeError: If color format is not supported
    """
    if isinstance(color, str):
        rgba = ImageColor.getrgb(color) + (255,)
        rgba = tuple([*rgba, *((255,) * (4 - len(rgba)))])
    elif isinstance(color, int):
        rgba = (color, color, color, 255)
    elif isinstance(color, (list, tuple)):
        rgba = tuple([*color, *((255,) * (4 - len(color)))])
    else:
        raise TypeError(f"Invalid color type: {type(color)}")

    return rgba


def _parse_color_to_mode(color, mode: Literal['RGB', 'RGBA', 'L', 'LA']):
    """
    Convert color to the specified image mode format.

    :param color: Color specification (string, integer, or tuple/list)
    :type color: Union[str, int, Tuple, list]
    :param mode: Target image mode ('RGB', 'RGBA', 'L', or 'LA')
    :type mode: Literal['RGB', 'RGBA', 'L', 'LA']

    :return: Color value in the specified mode format
    :rtype: Union[int, Tuple[int, int], Tuple[int, int, int], Tuple[int, int, int, int]]

    :raises ValueError: If the specified image mode is not supported
    """
    rgba = _parse_color_to_rgba(color)
    if mode == 'L':
        return int(0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2])
    elif mode == "LA":
        gray = int(0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2])
        return gray, rgba[3]
    elif mode == "RGB":
        return rgba[:3]
    elif mode == "RGBA":
        return rgba
    else:
        raise ValueError(f"Unsupported image mode: {mode!r}")


def pad_image_to_size(pic: ImageTyping, size: Union[int, Tuple[int, int]],
                      background_color: Union[str, int, Tuple[int, int, int], Tuple[int, int, int, int]] = 'white',
                      interpolation: int = Image.BILINEAR):
    """
    Resize and pad an image to the specified size while maintaining aspect ratio.

    The function first resizes the image to fit within the target dimensions while
    preserving the aspect ratio, then pads the result with the specified background
    color to reach the exact target size.

    :param pic: Input image (PIL Image, file path, or other supported format)
    :type pic: ImageTyping
    :param size: Target size as an integer or tuple of (width, height)
    :type size: Union[int, Tuple[int, int]]
    :param background_color: Color to use for padding (name, RGB tuple, etc.)
    :type background_color: Union[str, int, Tuple[int, int, int], Tuple[int, int, int, int]]
    :param interpolation: PIL interpolation method for resizing
    :type interpolation: int

    :return: Resized and padded image
    :rtype: PIL.Image.Image

    :example:

    >>> from PIL import Image
    >>> img = Image.new('RGB', (100, 50))
    >>> padded = pad_image_to_size(img, (200, 200), background_color='blue')
    >>> padded.size
    (200, 200)
    """
    pic = load_image(pic, force_background=None, mode=None)
    target_w, target_h = _parse_size(size)
    original_w, original_h = pic.size
    ratio = min(target_w / original_w, target_h / original_h)
    new_w, new_h = round(original_w * ratio), round(original_h * ratio)

    resized = pic.resize((new_w, new_h), interpolation)
    bg_color = _parse_color_to_mode(background_color, pic.mode)
    canvas = Image.new(pic.mode, (target_w, target_h), bg_color)
    canvas.paste(resized, ((target_w - new_w) // 2, (target_h - new_h) // 2))

    return canvas
