"""
This module provides utilities for handling image blob URLs, including conversion between images and blob URLs,
and validation of blob URL format.

The module supports:

- Converting images to blob URLs with specified formats
- Loading images from blob URLs
- Validating image blob URL format
- Handling various image formats and MIME types
"""

import base64
import re
import warnings
from io import BytesIO

from PIL import Image

from .image import load_image, ImageTyping

__all__ = [
    'to_blob_url',
    'load_image_from_blob_url',
    'is_valid_image_blob_url',
]

_FORMAT_REPLACE = {'JPG': 'JPEG'}


def to_blob_url(image: ImageTyping, format: str = 'jpg', **save_kwargs) -> str:
    """
    Convert an image to a blob URL string.

    :param image: The input image, can be PIL Image, numpy array, or file path
    :type image: ImageTyping
    :param format: The desired image format for the blob URL, defaults to 'jpg'
    :type format: str
    :param save_kwargs: Additional keyword arguments passed to PIL Image.save()
    :return: A blob URL string containing the encoded image data
    :rtype: str

    :example:
        >>> img = Image.open('test.jpg')
        >>> blob_url = to_blob_url(img, format='png', quality=95)
        >>> print(blob_url)  # data:image/png;base64,...</pre>
    """
    image = load_image(image, mode=None, force_background=None)
    format = (_FORMAT_REPLACE.get(format.upper(), format)).upper()
    with BytesIO() as buffer:
        image.save(buffer, **{'format': format, **save_kwargs})
        buffer.seek(0)
        mime_type = Image.MIME.get(format.upper(), f'image/{format.lower()}')
        base64_str = base64.b64encode(buffer.getvalue()).decode('ascii')
        return f"data:{mime_type};base64,{base64_str}"


def load_image_from_blob_url(blob_url: str) -> Image.Image:
    """
    Load an image from a blob URL string.

    :param blob_url: The blob URL string containing encoded image data
    :type blob_url: str
    :return: A PIL Image object
    :rtype: PIL.Image.Image
    :raises ValueError: If the blob URL uses an unsupported encoding method
    :warns UserWarning: If MIME type doesn't match the actual image format or is invalid

    :example:
        >>> blob_url = "data:image/png;base64,..."
        >>> img = load_image_from_blob_url(blob_url)
        >>> img.show()
    """
    header, data = blob_url.split(",", maxsplit=1)
    meta_parts = header.split(";")
    mimetype = meta_parts[0][5:]
    encoding = meta_parts[1] if len(meta_parts) > 1 else ""

    if encoding != "base64":
        raise ValueError(f'Unsupported blob encoding method - {encoding!r}.')

    decoded_data = base64.b64decode(data)
    with BytesIO(decoded_data) as buffer:
        image = Image.open(buffer)
        image.load()

        if '/' in mimetype:
            expected_type = mimetype.split('/')[-1].upper()
            actual_format = image.format.upper() if image.format else None
            if actual_format != expected_type:
                warnings.warn(
                    f"MIME type {mimetype!r} does not match detected image format {image.format!r}",
                    UserWarning
                )
        else:
            warnings.warn(
                f"Invalid MIME type format: {mimetype!r}",
                UserWarning
            )

        return image


_IMAGE_BLOB_URI_REGEX = re.compile(
    r'^data:image/'  # Required MIME type prefix
    r'[^;,]+'  # Image subtype (at least one character)
    r'(;[^;,]+)*'  # Optional parameters (like base64)
    r','  # Data separator
    r'.*',  # Data part (can be empty)
    flags=re.IGNORECASE  # Case-insensitive (e.g., DATA:IMAGE/PNG)
)


def is_valid_image_blob_url(blob_url: str) -> bool:
    """
    Efficiently validate the format of an image blob URL (without validating data content).

    :param blob_url: The URL string to validate
    :type blob_url: str
    :return: True if the string is a valid image blob URL, False otherwise
    :rtype: bool

    :example:
        Valid formats:

        >>> is_valid_image_blob_url('data:image/png;base64,ABC')  # True
        >>> is_valid_image_blob_url('data:image/svg+xml,<svg/>')  # True
        >>> is_valid_image_blob_url('DATA:IMAGE/JPEG;quality=95,...')  # True

        Invalid formats:

        >>> is_valid_image_blob_url('data:text/plain,hello')  # False
        >>> is_valid_image_blob_url('data:image/png')  # False
        >>> is_valid_image_blob_url('data:image/;base64,ABC')  # False
    """
    return bool(_IMAGE_BLOB_URI_REGEX.fullmatch(blob_url))
