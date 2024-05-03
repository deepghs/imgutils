from os import PathLike
from typing import Union, BinaryIO, List, Tuple, Optional

from PIL import Image

__all__ = [
    'ImageTyping', 'load_image',
    'MultiImagesTyping', 'load_images',
    'add_background_for_rgba',
]


def _is_readable(obj):
    return hasattr(obj, 'read') and hasattr(obj, 'seek')


ImageTyping = Union[str, PathLike, bytes, bytearray, BinaryIO, Image.Image]
MultiImagesTyping = Union[ImageTyping, List[ImageTyping], Tuple[ImageTyping, ...]]


def _has_alpha_channel(image: Image.Image) -> bool:
    return any(band in {'A', 'a', 'P'} for band in image.getbands())


def load_image(image: ImageTyping, mode=None, force_background: Optional[str] = 'white'):
    """
    Loads the image from the provided source and applies necessary transformations.

    The function supports loading images from various sources, such as file paths, binary data, or file-like objects.
    It opens the image using the PIL library and converts it to the specified mode if required.
    If the image has an RGBA (4-channel) format and a ``force_background`` value is provided, a background of
    the specified color will be added to avoid data anomalies during subsequent conversion processes.

    :param image: The source of the image to be loaded.
    :type image: Union[str, PathLike, bytes, bytearray, BinaryIO, Image.Image]

    :param mode: The mode to convert the image to. If None, the original mode will be retained. (default: ``None``)
    :type mode: str or None

    :param force_background: The color of the background to be added for RGBA images.
                             If None, no background will be added. (default: ``white``)
    :type force_background: str or None

    :return: The loaded and transformed image.
    :rtype: Image.Image
    """
    if isinstance(image, (str, PathLike, bytes, bytearray, BinaryIO)) or _is_readable(image):
        image = Image.open(image)
    elif isinstance(image, Image.Image):
        pass  # just do nothing
    else:
        raise TypeError(f'Unknown image type - {image!r}.')

    if _has_alpha_channel(image) and force_background is not None:
        image = add_background_for_rgba(image, force_background)

    if mode is not None and image.mode != mode:
        image = image.convert(mode)

    return image


def load_images(images: MultiImagesTyping, mode=None, force_background: Optional[str] = 'white') -> List[Image.Image]:
    """
    Loads a list of images from the provided sources and applies necessary transformations.

    The function takes a single image or a list/tuple of multiple images and calls :func:`load_image` function
    on each item to load and transform the images. The images are returned as a list of PIL Image objects.

    :param images: The sources of the images to be loaded.
    :type images: MultiImagesTyping

    :param mode: The mode to convert the images to. If None, the original modes will be retained. (default: ``None``)
    :type mode: str or None

    :param force_background: The color of the background to be added for RGBA images.
                             If None, no background will be added. (default: ``white``)
    :type force_background: str or None

    :return: A list of loaded and transformed images.
    :rtype: List[Image.Image]
    """
    if not isinstance(images, (list, tuple)):
        images = [images]

    return [load_image(item, mode, force_background) for item in images]


def add_background_for_rgba(image: ImageTyping, background: str = 'white'):
    """
    Adds a background color to the RGBA image if it has an alpha channel.

    The function checks if the provided image is in RGBA format and has an alpha channel.
    If it does, a background of the specified color will be added to the image using
    the :func:`imgutils.data.layer.istack` function. The resulting image is then converted to RGB.

    :param image: The RGBA image to add a background to.
    :type image: ImageTyping

    :param background: The color of the background to be added. (default: ``white``)
    :type background: str

    :return: The image with the added background, converted to RGB.
    :rtype: Image.Image
    """
    from .layer import istack
    return istack(background, image).convert('RGB')
