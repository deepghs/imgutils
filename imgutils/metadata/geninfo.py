"""
This module provides functions for reading and writing generation information (geninfo) to image files.
It supports different image formats including PNG, EXIF, and GIF.

The module includes functions for:

1. Reading geninfo from image parameters, EXIF data, and GIF comments
2. Writing geninfo to image parameters, EXIF data, and GIF comments

These functions are useful for storing and retrieving metadata about image generation,
particularly in the context of AI-generated images.
"""

from typing import Optional

import piexif
from PIL.PngImagePlugin import PngInfo
from piexif.helper import UserComment

from ..data import ImageTyping, load_image


def read_geninfo_parameters(image: ImageTyping) -> Optional[str]:
    """
    Read generation information from image parameters.

    :param image: The input image.
    :type image: ImageTyping

    :return: The generation information if found, None otherwise.
    :rtype: Optional[str]

    This function loads the image and attempts to retrieve the 'parameters' 
    information from the image metadata. It's commonly used for PNG images 
    where generation information is stored in the image parameters.
    """
    image = load_image(image, mode=None, force_background=None)
    infos = image.info or {}
    return infos.get('parameters')


def read_geninfo_exif(image: ImageTyping) -> Optional[str]:
    """
    Read generation information from EXIF data.

    :param image: The input image.
    :type image: ImageTyping

    :return: The generation information if found in EXIF data, None otherwise.
    :rtype: Optional[str]

    This function attempts to read generation information from the EXIF metadata
    of the image. It specifically looks for the UserComment field in the EXIF data.
    If the EXIF data is invalid or not present, it returns None.
    """
    image = load_image(image, mode=None, force_background=None)
    infos = image.info or {}
    if "exif" in infos:
        exif_data = infos["exif"]
        try:
            exif = piexif.load(exif_data)
        except OSError:
            # memory / exif was not valid so piexif tried to read from a file
            exif = None

        exif_comment = (exif or {}).get("Exif", {}).get(piexif.ExifIFD.UserComment, b"")
        try:
            exif_comment = UserComment.load(exif_comment)
        except ValueError:
            exif_comment = exif_comment.decode("utf8", errors="ignore")

        return exif_comment
    else:
        return None


def read_geninfo_gif(image: ImageTyping) -> Optional[str]:
    """
    Read generation information from GIF comment.

    :param image: The input image.
    :type image: ImageTyping

    :return: The generation information if found in GIF comment, None otherwise.
    :rtype: Optional[str]

    This function is specifically designed to read generation information from
    GIF images. It looks for the 'comment' field in the image metadata, which is
    commonly used in GIF files to store additional information.
    """
    image = load_image(image, mode=None, force_background=None)
    infos = image.info or {}
    if "comment" in infos and isinstance(infos['comment'], (bytes, bytearray)):  # for gif
        return infos["comment"].decode("utf8", errors="ignore")
    else:
        return None


def write_geninfo_parameters(image: ImageTyping, dst_filename: str, geninfo: str, **kwargs):
    """
    Write generation information to image parameters.

    :param image: The input image.
    :type image: ImageTyping
    :param dst_filename: The destination filename to save the image with geninfo.
    :type dst_filename: str
    :param geninfo: The generation information to write.
    :type geninfo: str
    :param kwargs: Additional keyword arguments to pass to the image save function.

    This function writes the provided generation information to the image parameters.
    It's commonly used for PNG images where generation information can be stored in
    the image metadata. The function creates a PngInfo object, adds the geninfo as
    'parameters', and saves the image with this metadata.
    """
    pnginfo = PngInfo()
    pnginfo.add_text('parameters', geninfo)

    image = load_image(image, force_background=None, mode=None)
    image.save(dst_filename, pnginfo=pnginfo, **kwargs)


def write_geninfo_exif(image: ImageTyping, dst_filename: str, geninfo: str, **kwargs):
    """
    Write generation information to EXIF data.

    :param image: The input image.
    :type image: ImageTyping
    :param dst_filename: The destination filename to save the image with geninfo.
    :type dst_filename: str
    :param geninfo: The generation information to write.
    :type geninfo: str
    :param kwargs: Additional keyword arguments to pass to the image save function.

    This function writes the provided generation information to the EXIF metadata
    of the image. It creates an EXIF dictionary with the geninfo stored in the
    UserComment field, converts it to bytes, and saves the image with this EXIF data.
    """
    exif_dict = {
        "Exif": {piexif.ExifIFD.UserComment: UserComment.dump(geninfo, encoding="unicode")}}
    exif_bytes = piexif.dump(exif_dict)

    image = load_image(image, force_background=None, mode=None)
    image.save(dst_filename, exif=exif_bytes, **kwargs)


def write_geninfo_gif(image: ImageTyping, dst_filename: str, geninfo: str, **kwargs):
    """
    Write generation information to GIF comment.

    :param image: The input image.
    :type image: ImageTyping
    :param dst_filename: The destination filename to save the image with geninfo.
    :type dst_filename: str
    :param geninfo: The generation information to write.
    :type geninfo: str
    :param kwargs: Additional keyword arguments to pass to the image save function.

    This function is specifically designed to write generation information to
    GIF images. It adds the geninfo to the image's 'comment' field, which is
    a standard way of including metadata in GIF files.
    """
    image = load_image(image, force_background=None, mode=None)
    image.info['comment'] = geninfo.encode('utf-8')
    image.save(dst_filename, **kwargs)
