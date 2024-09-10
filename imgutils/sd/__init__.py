"""
Overview:

    Utilities for dealing with data from AUTOMATIC1111/stable-diffusion-webui and NovelAI generated images.

    This module provides a collection of utilities for handling metadata and image processing
    related to Stable Diffusion WebUI (SDWUI) and NovelAI generated images. It includes
    functions for parsing and extracting metadata, reading and writing model metadata,
    and manipulating NovelAI image metadata.

Submodules:

    - metadata: Functions for parsing and extracting Stable Diffusion metadata.
    - model: Utilities for reading and writing metadata from/to model files.
    - nai: Functions for handling NovelAI image metadata.

For detailed usage of each function and class, please refer to their individual docstrings.
"""

from .metadata import parse_sdmeta_from_text, get_sdmeta_from_image, SDMetaData, save_image_with_sdmeta
from .model import read_metadata, save_with_metadata
from .nai import get_naimeta_from_image, NAIMetaData, add_naimeta_to_image, save_image_with_naimeta, NAIMetadata
