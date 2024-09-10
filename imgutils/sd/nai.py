"""
This module provides functionality for handling metadata in images, specifically for the Novel AI (NAI) format.
It includes classes and functions for extracting, creating, and saving NAI metadata in images.

The module offers the following main features:

1. Extraction of NAI metadata from images
2. Creation of NAI metadata objects
3. Adding NAI metadata to images
4. Saving images with NAI metadata

This module is particularly useful for working with AI-generated images and their associated metadata.
"""

import json
import mimetypes
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Union

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from ..data import load_image, ImageTyping
from ..metadata import read_lsb_metadata, write_lsb_metadata, LSBReadError, read_geninfo_parameters, \
    read_geninfo_exif, read_geninfo_gif, write_geninfo_exif, write_geninfo_gif

mimetypes.add_type('image/webp', '.webp')


@dataclass
class NAIMetadata:
    """
    A dataclass representing Novel AI (NAI) metadata.

    This class encapsulates various metadata fields associated with NAI-generated images.

    :param software: The software used to generate the image.
    :type software: str
    :param source: The source of the image.
    :type source: str
    :param parameters: A dictionary containing generation parameters.
    :type parameters: dict
    :param title: The title of the image (optional).
    :type title: Optional[str]
    :param generation_time: The time taken to generate the image (optional).
    :type generation_time: Optional[float]
    :param description: A description of the image (optional).
    :type description: Optional[str]
    """

    software: str
    source: str
    parameters: dict
    title: Optional[str] = None
    generation_time: Optional[float] = None
    description: Optional[str] = None

    @property
    def json(self) -> dict:
        """
        Convert the NAIMetadata to a JSON-compatible dictionary.

        :return: A dictionary representation of the metadata.
        :rtype: dict
        """
        data = {
            'Software': self.software,
            'Source': self.source,
            'Comment': json.dumps(self.parameters),
        }
        if self.title is not None:
            data['Title'] = self.title
        if self.generation_time is not None:
            data['Generation time'] = json.dumps(self.generation_time)
        if self.description is not None:
            data['Description'] = self.description
        return data

    @property
    def pnginfo(self) -> PngInfo:
        """
        Convert the NAIMetadata to a PngInfo object.

        This property creates a PngInfo object with the metadata information,
        which can be used when saving PNG images.

        :return: A PngInfo object containing the metadata.
        :rtype: PngInfo
        """
        info = PngInfo()
        for key, value in self.json.items():
            info.add_text(key, value)
        return info


class _InvalidNAIMetaError(Exception):
    """
    Custom exception raised when NAI metadata is invalid.
    """
    pass


def _naimeta_validate(data):
    """
    Validate the NAI metadata.

    :param data: The metadata to validate.
    :type data: dict
    :return: The validated metadata.
    :rtype: dict
    :raises _InvalidNAIMetaError: If the metadata is invalid.
    """
    if isinstance(data, dict) and data.get('Software') and data.get('Source') and data.get('Comment'):
        return data
    else:
        raise _InvalidNAIMetaError


def _get_naimeta_raw(image: ImageTyping) -> dict:
    """
    Extract raw NAI metadata from an image.

    This function attempts to extract metadata from the image using various methods,
    including LSB extraction, image info dictionary, and other specific metadata formats.

    :param image: The input image.
    :type image: ImageTyping
    :return: A dictionary containing the raw metadata.
    :rtype: dict
    :raises _InvalidNAIMetaError: If no valid metadata is found.
    """
    image = load_image(image, force_background=None, mode=None)
    try:
        return _naimeta_validate(read_lsb_metadata(image))
    except (LSBReadError, _InvalidNAIMetaError):
        pass

    try:
        return _naimeta_validate(image.info or {})
    except (LSBReadError, _InvalidNAIMetaError):
        pass

    try:
        return _naimeta_validate(json.loads(read_geninfo_parameters(image)))
    except (TypeError, json.JSONDecodeError, _InvalidNAIMetaError):
        pass

    try:
        return _naimeta_validate(json.loads(read_geninfo_exif(image)))
    except (TypeError, json.JSONDecodeError, _InvalidNAIMetaError):
        pass

    try:
        return _naimeta_validate(json.loads(read_geninfo_gif(image)))
    except (TypeError, json.JSONDecodeError, _InvalidNAIMetaError):
        raise _InvalidNAIMetaError


def get_naimeta_from_image(image: ImageTyping) -> Optional[NAIMetadata]:
    """
    Extract and create a NAIMetadata object from an image.

    This function attempts to extract NAI metadata from the given image and create a NAIMetadata object.
    If the required metadata fields are not present, it returns None.

    :param image: The input image.
    :type image: ImageTyping
    :return: A NAIMetadata object if successful, None otherwise.
    :rtype: Optional[NAIMetadata]
    """
    try:
        data = _get_naimeta_raw(image)
    except _InvalidNAIMetaError:
        return None
    else:
        return NAIMetadata(
            software=data['Software'],
            source=data['Source'],
            parameters=json.loads(data['Comment']),
            title=data.get('Title'),
            generation_time=float(data['Generation time']) if data.get('Generation time') else None,
            description=data.get('Description'),
        )


def add_naimeta_to_image(image: ImageTyping, metadata: NAIMetadata) -> Image.Image:
    """
    Add NAI metadata to an image using LSB (Least Significant Bit) encoding.

    :param image: The input image.
    :type image: ImageTyping
    :param metadata: The NAIMetadata object to add to the image.
    :type metadata: NAIMetadata
    :return: The image with added metadata.
    :rtype: Image.Image
    """
    image = load_image(image, mode=None, force_background=None)
    return write_lsb_metadata(image, data=metadata.pnginfo)


def _save_png_with_naimeta(image: Image.Image, dst_file: Union[str, os.PathLike], metadata: NAIMetadata, **kwargs):
    """
    Save a PNG image with NAI metadata.

    :param image: The image to save.
    :type image: Image.Image
    :param dst_file: The destination file path.
    :type dst_file: Union[str, os.PathLike]
    :param metadata: The NAIMetadata object to include in the image.
    :type metadata: NAIMetadata
    :param kwargs: Additional keyword arguments for image saving.
    """
    image.save(dst_file, pnginfo=metadata.pnginfo, **kwargs)


def _save_exif_with_naimeta(image: Image.Image, dst_file: Union[str, os.PathLike], metadata: NAIMetadata, **kwargs):
    """
    Save an image with NAI metadata in EXIF format.

    :param image: The image to save.
    :type image: Image.Image
    :param dst_file: The destination file path.
    :type dst_file: Union[str, os.PathLike]
    :param metadata: The NAIMetadata object to include in the image.
    :type metadata: NAIMetadata
    :param kwargs: Additional keyword arguments for image saving.
    """
    write_geninfo_exif(image, dst_file, json.dumps(metadata.json), **kwargs)


def _save_gif_with_naimeta(image: Image.Image, dst_file: Union[str, os.PathLike], metadata: NAIMetadata, **kwargs):
    """
    Save a GIF image with NAI metadata.

    :param image: The image to save.
    :type image: Image.Image
    :param dst_file: The destination file path.
    :type dst_file: Union[str, os.PathLike]
    :param metadata: The NAIMetadata object to include in the image.
    :type metadata: NAIMetadata
    :param kwargs: Additional keyword arguments for image saving.
    """
    write_geninfo_gif(image, dst_file, json.dumps(metadata.json), **kwargs)


_FN_IMG_SAVE = {
    'image/png': _save_png_with_naimeta,
    'image/jpeg': _save_exif_with_naimeta,
    'image/webp': _save_exif_with_naimeta,
    'image/gif': _save_gif_with_naimeta,
}
_LSB_ALLOWED_TYPES = {'image/png', 'image/tiff'}


def save_image_with_naimeta(
        image: ImageTyping, dst_file: Union[str, os.PathLike], metadata: NAIMetadata,
        add_lsb_meta: Union[str, bool] = 'auto', save_metainfo: Union[str, bool] = 'auto', **kwargs) -> Image.Image:
    """
    Save an image with NAI metadata.

    This function saves the given image with the provided NAI metadata. It can add LSB metadata
    and save metainfo based on the image format and user preferences.

    :param image: The input image.
    :type image: ImageTyping
    :param dst_file: The destination file path.
    :type dst_file: Union[str, os.PathLike]
    :param metadata: The NAIMetadata object to include in the image.
    :type metadata: NAIMetadata
    :param add_lsb_meta: Whether to add LSB metadata. Can be 'auto', True, or False.
    :type add_lsb_meta: Union[str, bool]
    :param save_metainfo: Whether to save metainfo. Can be 'auto', True, or False.
    :type save_metainfo: Union[str, bool]
    :param kwargs: Additional keyword arguments for image saving.
    :return: The saved image.
    :rtype: Image.Image
    :raises ValueError: If LSB metadata cannot be saved to the specified image format.
    :raises SystemError: If the image format is not supported for saving metainfo.
    """
    mimetype, _ = mimetypes.guess_type(dst_file)
    if add_lsb_meta == 'auto':
        if mimetype in _LSB_ALLOWED_TYPES:
            add_lsb_meta = True
        else:
            add_lsb_meta = False
    else:
        if add_lsb_meta and mimetype not in _LSB_ALLOWED_TYPES:
            raise ValueError('LSB metadata cannot be saved to lossy image format or RGBA-incompatible format, '
                             'add_lsb_meta will be disabled. '
                             f'Only {", ".join(sorted(_LSB_ALLOWED_TYPES))} images supported.')
    if save_metainfo == 'auto':
        if mimetype in _FN_IMG_SAVE:
            save_metainfo = True
        else:
            save_metainfo = False
    else:
        if save_metainfo and mimetype not in _FN_IMG_SAVE:
            raise SystemError(f'Not supported to save as a {mimetype!r} type, '
                              f'supported mimetypes are {sorted(_FN_IMG_SAVE.keys())!r}.')
    if not add_lsb_meta and not save_metainfo:
        warnings.warn(f'Both LSB meta and pnginfo is disabled, no metadata will be saved to {dst_file!r}.')

    image = load_image(image, mode=None, force_background=None)
    if add_lsb_meta:
        image = add_naimeta_to_image(image, metadata=metadata)
    if save_metainfo:
        _FN_IMG_SAVE[mimetype](image, dst_file, metadata, **kwargs)
    else:
        image.save(dst_file, **kwargs)
    return image
