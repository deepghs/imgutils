"""
This module provides utility functions for image processing and manipulation using the PIL (Python Imaging Library) library.

It includes functions for loading images from various sources, handling multiple images, adding backgrounds to RGBA images,
and checking for alpha channels. The module is designed to simplify common image-related tasks in Python applications.

Key features:
- Loading images from different sources (file paths, binary data, file-like objects)
- Handling multiple images at once
- Adding backgrounds to RGBA images
- Checking for alpha channels in images

This module is particularly useful for applications that require image preprocessing or manipulation before further processing or analysis.
"""

from os import PathLike
from typing import Union, BinaryIO, List, Tuple, Optional

from PIL import Image

__all__ = [
    'ImageTyping',
    'load_image',
    'MultiImagesTyping',
    'load_images',
    'add_background_for_rgba',
    'has_alpha_channel',
]


def _is_readable(obj):
    """
    Check if an object is readable (has 'read' and 'seek' methods).

    :param obj: The object to check for readability.
    :type obj: Any

    :return: True if the object is readable, False otherwise.
    :rtype: bool
    """
    return hasattr(obj, 'read') and hasattr(obj, 'seek')


ImageTyping = Union[str, PathLike, bytes, bytearray, BinaryIO, Image.Image]
MultiImagesTyping = Union[ImageTyping, List[ImageTyping], Tuple[ImageTyping, ...]]


def has_alpha_channel(image: Image.Image) -> bool:
    """
    Determine if the given Pillow image object has an alpha channel (transparency)

    :param image: Pillow image object
    :type image: Image.Image

    :return: Boolean, True if it has an alpha channel, False otherwise
    :rtype: bool
    """
    # Get the image mode
    mode = image.mode

    # Modes that directly include an alpha channel
    if mode in ('RGBA', 'LA', 'PA'):
        return True

    if getattr(image, 'palette'):
        # Check if there's a transparent palette
        try:
            image.palette.getcolor((0, 0, 0, 0))
            return True  # cannot find a line to trigger this
        except ValueError:
            pass

    # For other modes, check if 'transparency' key exists in image info
    return 'transparency' in image.info


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

    :raises TypeError: If the provided image type is not supported.

    :example:
    >>> from PIL import Image
    >>> img = load_image('path/to/image.png', mode='RGB', force_background='white')
    >>> isinstance(img, Image.Image)
    True
    >>> img.mode
    'RGB'
    """
    if isinstance(image, (str, PathLike, bytes, bytearray, BinaryIO)) or _is_readable(image):
        image = Image.open(image)
    elif isinstance(image, Image.Image):
        pass  # just do nothing
    else:
        raise TypeError(f'Unknown image type - {image!r}.')

    if has_alpha_channel(image) and force_background is not None:
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

    :example:
    >>> img_paths = ['path/to/image1.png', 'path/to/image2.jpg']
    >>> loaded_images = load_images(img_paths, mode='RGB')
    >>> len(loaded_images)
    2
    >>> all(isinstance(img, Image.Image) for img in loaded_images)
    True
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

    :example:
    >>> from PIL import Image
    >>> rgba_image = Image.new('RGBA', (100, 100), (255, 0, 0, 128))
    >>> rgb_image = add_background_for_rgba(rgba_image, background='blue')
    >>> rgb_image.mode
    'RGB'
    """
    image = load_image(image, force_background=None, mode=None)
    try:
        ret_image = Image.new('RGBA', image.size, background)
        ret_image.paste(image, (0, 0), mask=image)
    except ValueError:
        ret_image = image
    if ret_image.mode != 'RGB':
        ret_image = ret_image.convert('RGB')
    return ret_image
