import json
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Union

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from .extract import ImageLsbDataExtractor
from .inject import inject_data
from ...data import load_image, ImageTyping


@dataclass
class NAIMetadata:
    software: str
    source: str
    parameters: dict
    title: Optional[str] = None
    generation_time: Optional[float] = None
    description: Optional[str] = None

    @property
    def pnginfo(self) -> PngInfo:
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
    image = load_image(image, force_background=None, mode=None)
    try:
        return ImageLsbDataExtractor().extract_data(image)
    except (ValueError, json.JSONDecodeError):
        return image.info or {}


def get_naimeta_from_image(image: ImageTyping) -> Optional[NAIMetadata]:
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
    if isinstance(metadata, NAIMetadata):
        pnginfo = metadata.pnginfo
    elif isinstance(metadata, PngInfo):
        pnginfo = metadata
    else:
        raise TypeError(f'Unknown metadata type for NAI - {metadata!r}.')  # pragma: no cover
    return pnginfo


def add_naimeta_to_image(image: ImageTyping, metadata: Union[NAIMetadata, PngInfo]) -> Image.Image:
    pnginfo = _get_pnginfo(metadata)
    image = load_image(image, mode=None, force_background=None)
    return inject_data(image, data=pnginfo)


def save_image_with_naimeta(image: ImageTyping, dst_file: Union[str, os.PathLike],
                            metadata: Union[NAIMetadata, PngInfo],
                            add_lsb_meta: bool = True, save_pnginfo: bool = True, **kwargs) -> Image.Image:
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
