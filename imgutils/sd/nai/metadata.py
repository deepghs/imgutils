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
import os
import warnings
import zlib
from dataclasses import dataclass
from typing import Optional, Union

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from .extract import ImageLsbDataExtractor
from .inject import inject_data
from ...data import load_image, ImageTyping


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
    def pnginfo(self) -> PngInfo:
        """
        Convert the NAIMetadata to a PngInfo object.

        This property creates a PngInfo object with the metadata information,
        which can be used when saving PNG images.

        :return: A PngInfo object containing the metadata.
        :rtype: PngInfo
        """
        info = PngInfo()
        info.add_text('Software', self.software)
        info.add_text('Source', self.source)
        if self.title is not None:
            info.add_text('Title', self.title)
        if self.generation_time is not None:
            info.add_text('Generation time', json.dumps(self.generation_time)),
        if self.description is not None:
            info.add_text('Description', self.description)
        if self.parameters is not None:
            info.add_text('Comment', json.dumps(self.parameters))
        return info


def _get_naimeta_raw(image: ImageTyping) -> dict:
    """
    Extract raw NAI metadata from an image.

    This function attempts to extract metadata from the image using LSB (Least Significant Bit) extraction.
    If that fails, it falls back to using the image's info dictionary.

    :param image: The input image.
    :type image: ImageTyping

    :return: A dictionary containing the raw metadata.
    :rtype: dict
    """
    image = load_image(image, force_background=None, mode=None)
    try:
        return ImageLsbDataExtractor().extract_data(image)
    except (ValueError, json.JSONDecodeError, zlib.error, OSError, UnicodeDecodeError):
        # ValueError: binary data with wrong format
        # json.JSONDecodeError: zot a json-formatted data
        # zlib.error, OSError: not zlib compressed binary data
        # UnicodeDecodeError: cannot decode as utf-8 text
        return image.info or {}


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
    data = _get_naimeta_raw(image)
    if data.get('Software') and data.get('Source') and data.get('Comment'):
        return NAIMetadata(
            software=data['Software'],
            source=data['Source'],
            parameters=json.loads(data['Comment']),
            title=data.get('Title'),
            generation_time=float(data['Generation time']) if data.get('Generation time') else None,
            description=data.get('Description'),
        )
    else:
        return None


def _get_pnginfo(metadata: Union[NAIMetadata, PngInfo]) -> PngInfo:
    """
    Convert metadata to PngInfo object.

    This function takes either a NAIMetadata object or a PngInfo object and returns a PngInfo object.

    :param metadata: The metadata to convert.
    :type metadata: Union[NAIMetadata, PngInfo]

    :return: A PngInfo object.
    :rtype: PngInfo

    :raises TypeError: If the metadata is neither NAIMetadata nor PngInfo.
    """
    if isinstance(metadata, NAIMetadata):
        pnginfo = metadata.pnginfo
    elif isinstance(metadata, PngInfo):
        pnginfo = metadata
    else:
        raise TypeError(f'Unknown metadata type for NAI - {metadata!r}.')  # pragma: no cover
    return pnginfo


def add_naimeta_to_image(image: ImageTyping, metadata: Union[NAIMetadata, PngInfo]) -> Image.Image:
    """
    Add NAI metadata to an image.

    This function injects the provided metadata into the image using LSB injection.

    :param image: The input image.
    :type image: ImageTyping
    :param metadata: The metadata to add to the image.
    :type metadata: Union[NAIMetadata, PngInfo]

    :return: The image with added metadata.
    :rtype: Image.Image
    """
    pnginfo = _get_pnginfo(metadata)
    image = load_image(image, mode=None, force_background=None)
    return inject_data(image, data=pnginfo)


def save_image_with_naimeta(image: ImageTyping, dst_file: Union[str, os.PathLike],
                            metadata: Union[NAIMetadata, PngInfo],
                            add_lsb_meta: bool = True, save_pnginfo: bool = True, **kwargs) -> Image.Image:
    """
    Save an image with NAI metadata.

    This function saves the given image to a file, optionally adding NAI metadata using LSB injection
    and/or saving it as PNG metadata.

    :param image: The input image.
    :type image: ImageTyping
    :param dst_file: The destination file path.
    :type dst_file: Union[str, os.PathLike]
    :param metadata: The metadata to add to the image.
    :type metadata: Union[NAIMetadata, PngInfo]
    :param add_lsb_meta: Whether to add metadata using LSB injection. Defaults to True.
    :type add_lsb_meta: bool
    :param save_pnginfo: Whether to save metadata as PNG metadata. Defaults to True.
    :type save_pnginfo: bool
    :param kwargs: Additional keyword arguments to pass to the image save function.

    :return: The saved image.
    :rtype: Image.Image

    :raises Warning: If both LSB meta and pnginfo are disabled.
    """
    pnginfo = _get_pnginfo(metadata)
    image = load_image(image, mode=None, force_background=None)
    if not add_lsb_meta and not save_pnginfo:
        warnings.warn(f'Both LSB meta and pnginfo is disabled, no metadata will be saved to {dst_file!r}.')
    if add_lsb_meta:
        image = add_naimeta_to_image(image, metadata=pnginfo)
    if save_pnginfo:
        kwargs['pnginfo'] = pnginfo
    image.save(dst_file, **kwargs)
    return image
