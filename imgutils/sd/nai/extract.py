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

import numpy as np
from PIL import Image


class LSBExtractor(object):
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

    def get_one_byte(self):
        """
        Extract and return one byte of data.

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

        :return: The extracted 32-bit integer, or None if not enough data is available.
        :rtype: int or None
        """
        bytes_list = self.get_next_n_bytes(4)
        if len(bytes_list) == 4:
            integer_value = int.from_bytes(bytes_list, byteorder='big')
            return integer_value
        else:
            return None


class ImageLsbDataExtractor(object):
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

    def extract_data(self, image: Image.Image) -> dict:
        """
        Extract hidden JSON data from the given image.

        This method reads the LSB data from the image, verifies the magic number,
        and extracts, decompresses, and decodes the hidden JSON data.

        :param image: The input image.
        :type image: Image.Image
        :return: The extracted JSON data as a dictionary.
        :rtype: dict
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

        read_len = reader.read_32bit_integer() // 8
        json_data = reader.get_next_n_bytes(read_len)

        json_data = json.loads(gzip.decompress(json_data).decode("utf-8"))
        return json_data
