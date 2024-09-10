"""
This module provides functionality for extracting hidden metadata from images using
LSB (Least Significant Bit) steganography.

It includes two main classes:

1. LSBExtractor: Extracts bits and bytes from image data.
2. ImageLsbDataExtractor: Uses LSBExtractor to extract and decode hidden JSON data from images.

The module is based on the implementation from the NovelAI project (https://github.com/NovelAI/novelai-image-metadata).

Usage:
    >>> from PIL import Image
    >>>
    >>> # Load an image
    >>> image = Image.open('path_to_image.png')
    >>>
    >>> # Create an extractor
    >>> extractor = ImageLsbDataExtractor()
    >>>
    >>> # Extract metadata
    >>> metadata = extractor.extract_data(image)
    >>>
    >>> # Process the extracted metadata
    >>> print(metadata)
"""

import gzip
import json
import zlib

import numpy as np
from PIL import Image

from imgutils.data import ImageTyping
from ...data import load_image


class LSBExtractor:
    """
    A class for extracting data hidden in the least significant bits of image pixels.

    This class provides methods to extract individual bits, bytes, and multi-byte values
    from image data using LSB steganography techniques.

    :param data: The image data as a numpy array.
    :type data: np.ndarray
    """

    def __init__(self, data: np.ndarray):
        """
        Initialize the LSBExtractor with image data.

        :param data: The image data as a numpy array.
        :type data: np.ndarray
        """
        self.data = data
        self.rows, self.cols, self.dim = data.shape
        self.bits = 0
        self.byte = 0
        self.row = 0
        self.col = 0

    def _extract_next_bit(self):
        """
        Extract the next bit from the image data.

        This method updates the internal state of the extractor,
        moving to the next pixel as necessary.

        :raises IOError: If there are no more bits to extract.
        """
        if self.row < self.rows and self.col < self.cols:
            bit = self.data[self.row, self.col, self.dim - 1] & 1
            self.bits += 1
            self.byte <<= 1
            self.byte |= bit
            self.row += 1
            if self.row == self.rows:
                self.row = 0
                self.col += 1
        else:
            raise IOError('Cannot read more bits.')

    def get_one_byte(self):
        """
        Extract and return one byte of data.

        This method extracts 8 bits from the image data to form a single byte.

        :return: A single byte of extracted data.
        :rtype: bytearray
        """
        while self.bits < 8:
            self._extract_next_bit()
        byte = bytearray([self.byte])
        self.bits = 0
        self.byte = 0
        return byte

    def get_next_n_bytes(self, n):
        """
        Extract and return the next n bytes of data.

        This method extracts multiple bytes from the image data.

        :param n: The number of bytes to extract.
        :type n: int
        :return: The extracted bytes.
        :rtype: bytearray
        """
        bytes_list = bytearray()
        for _ in range(n):
            byte = self.get_one_byte()
            if not byte:
                break
            bytes_list.extend(byte)
        return bytes_list

    def read_32bit_integer(self):
        """
        Extract and return a 32-bit integer from the image data.

        This method reads 4 bytes and interprets them as a big-endian 32-bit integer.

        :return: The extracted 32-bit integer, or None if not enough data is available.
        :rtype: int or None
        """
        bytes_list = self.get_next_n_bytes(4)
        if len(bytes_list) == 4:
            integer_value = int.from_bytes(bytes_list, byteorder='big')
            return integer_value
        else:
            return None


class ImageLsbDataExtractor:
    """
    A class for extracting hidden JSON data from images using LSB steganography.

    This class uses the LSBExtractor to read hidden data from an image,
    expecting a specific magic number and format for the hidden data.

    :param magic: The magic string used to identify the start of the hidden data.
    :type magic: str
    """

    def __init__(self, magic: str = "stealth_pngcomp"):
        """
        Initialize the ImageLsbDataExtractor with a magic string.

        :param magic: The magic string used to identify the start of the hidden data.
        :type magic: str
        """
        self._magic_bytes = magic.encode('utf-8')

    def extract_data(self, image: Image.Image) -> bytes:
        """
        Extract hidden data from the given image.

        This method checks for the magic number, reads the length of the hidden data,
        and then extracts the data.

        :param image: The image to extract data from.
        :type image: Image.Image
        :return: The extracted raw data.
        :rtype: bytes
        :raises ValueError: If the image is not in RGBA mode or if the magic number doesn't match.
        """
        if image.mode != 'RGBA':
            raise ValueError(f'Image should be in RGBA mode, but {image.mode!r} found.')
        # noinspection PyTypeChecker
        image = np.array(image)
        reader = LSBExtractor(image)

        read_magic = reader.get_next_n_bytes(len(self._magic_bytes))
        if not (self._magic_bytes == read_magic):
            raise ValueError(f'Image magic number mismatch, '
                             f'{self._magic_bytes!r} expected but {read_magic!r}.')

        next_int = reader.read_32bit_integer()
        if next_int is None:
            raise ValueError('No next int32 to read.')
        read_len = next_int // 8
        raw_data = reader.get_next_n_bytes(read_len)
        return raw_data


class LSBReadError(Exception):
    """
    Custom exception class for LSB reading errors.

    This exception is raised when there's an error during the LSB data extraction process.

    :param err: The original exception that caused the LSB read error.
    :type err: Exception
    """

    def __init__(self, err: Exception):
        """
        Initialize the LSBReadError with the original exception.

        :param err: The original exception that caused the LSB read error.
        :type err: Exception
        """
        Exception.__init__(self, (f'LSB Read Error - {err!r}', err))
        self.error = err


def read_lsb_raw_bytes(image: ImageTyping) -> bytes:
    """
    Read raw bytes of LSB-encoded data from an image.

    This function loads the image and uses ImageLsbDataExtractor to extract the hidden data.

    :param image: The image to extract data from. Can be a file path, URL, or Image object.
    :type image: ImageTyping
    :return: The extracted raw data.
    :rtype: bytes
    :raises LSBReadError: If there's an error during the extraction process.
    """
    image = load_image(image, mode=None, force_background=None)
    try:
        return ImageLsbDataExtractor().extract_data(image)
    except (ValueError, OSError, IOError, EOFError) as err:
        # ValueError: binary data with wrong format
        # IOError, EOFError: unable to read more from images
        # UnicodeDecodeError: cannot decode as utf-8 text
        raise LSBReadError(err)


def read_lsb_metadata(image: ImageTyping):
    """
    Read and decode LSB-encoded metadata from an image.

    This function extracts the raw bytes, decompresses them using gzip,
    and then decodes the result as a JSON object.

    :param image: The image to extract metadata from. Can be a file path, URL, or Image object.
    :type image: ImageTyping
    :return: The decoded metadata as a Python object.
    :rtype: dict
    :raises LSBReadError: If there's an error during the extraction or decoding process.
    """
    try:
        raw_data = read_lsb_raw_bytes(image)
        return json.loads(gzip.decompress(raw_data).decode("utf-8"))
    except (json.JSONDecodeError, zlib.error, gzip.BadGzipFile, EOFError, UnicodeDecodeError) as err:
        # zlib.error, gzip.BadGzipFile: unable to decompress via zlib method
        # json.JSONDecodeError, EOFError: not a json-formatted data
        # UnicodeDecodeError: cannot decode as utf-8 text
        raise LSBReadError(err)
