"""
Overview:
    Utilities for dealing with data from `AUTOMATIC1111/stable-diffusion-webui <https://github.com/AUTOMATIC1111/stable-diffusion-webui>`_.
"""
from .metadata import parse_sdmeta_from_text, get_sdmeta_from_image, SDMetaData
from .model import read_metadata, save_with_metadata
