"""
This module provides functionality for handling LSB (Least Significant Bit) data extraction and injection,
as well as managing Novel AI (NAI) metadata in images.

The module includes the following main components:

1. LSB extraction from images
2. Data injection into images
3. NAI metadata handling (extraction, creation, addition, and saving)

This module is particularly useful for working with steganography in images and
managing metadata for AI-generated images.
"""

from .extract import LSBExtractor, ImageLsbDataExtractor
from .inject import serialize_metadata, inject_data
from .metadata import get_naimeta_from_image, NAIMetadata, add_naimeta_to_image, save_image_with_naimeta
