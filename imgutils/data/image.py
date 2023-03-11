from os import PathLike
from typing import Union, BinaryIO, List, Tuple

from PIL import Image


def _is_readable(obj):
    return hasattr(obj, 'read') and hasattr(obj, 'seek')


ImageTyping = Union[str, PathLike, bytes, bytearray, BinaryIO, Image.Image]
MultiImagesTyping = Union[ImageTyping, List[ImageTyping], Tuple[ImageTyping, ...]]


def load_image(image: ImageTyping, mode=None):
    if isinstance(image, (str, PathLike, bytes, bytearray, BinaryIO)) or _is_readable(image):
        image = Image.open(image)
    elif isinstance(image, Image.Image):
        pass  # just do nothing
    else:
        raise TypeError(f'Unknown image type - {image!r}.')

    if mode is not None and image.mode != mode:
        image = image.convert(mode)

    return image


def load_images(images: MultiImagesTyping, mode=None) -> List[Image.Image]:
    if not isinstance(images, (list, tuple)):
        images = [images]

    return [load_image(item, mode) for item in images]
