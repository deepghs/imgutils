"""
This module provides functionality for reading and writing generation information (geninfo)
and metadata using various methods.

It includes functions for handling generation information in different formats (parameters, EXIF, GIF)
and working with LSB (Least Significant Bit) metadata.

The module is designed to be used in image processing and generation tasks, particularly in the context
of AI-generated images.
"""

from .geninfo import read_geninfo_parameters, read_geninfo_exif, read_geninfo_gif, \
    write_geninfo_parameters, write_geninfo_exif, write_geninfo_gif
from .lsb import read_lsb_raw_bytes, read_lsb_metadata, write_lsb_raw_bytes, write_lsb_metadata, LSBReadError
