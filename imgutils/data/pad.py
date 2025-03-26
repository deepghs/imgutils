from typing import Union, Tuple, Literal

from PIL import ImageColor, Image

from .image import ImageTyping, load_image

__all__ = [
    'pad_image_to_size',
]


def _parse_size(size):
    if isinstance(size, int):
        return size, size
    elif isinstance(size, (list, tuple)) and len(size) == 2:
        return int(size[0]), int(size[1])
    else:
        raise TypeError("Size must be int or tuple of two ints")


def _parse_color_to_rgba(color):
    if isinstance(color, str):
        rgba = ImageColor.getrgb(color) + (255,)
        rgba = tuple([*rgba, *((255,) * (4 - len(rgba)))])
    elif isinstance(color, int):
        rgba = (color, color, color, 255)
    elif isinstance(color, (list, tuple)):
        rgba = color + (255,) * (4 - len(color))
    else:
        raise TypeError(f"Invalid color type: {type(color)}")

    return rgba


def _parse_color_to_mode(color, mode: Literal['RGB', 'RGBA', 'L', 'LA']):
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
