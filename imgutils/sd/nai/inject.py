"""
This module provides functionality for injecting metadata and error correction codes into PNG images.

It includes classes and functions for:

- Bit shuffling and error correction encoding
- LSB (Least Significant Bit) injection of data into image pixels
- Serializing PNG metadata
- Injecting encoded data and metadata into PNG images

The module uses techniques like BCH error correction, bit manipulation, and LSB steganography
to embed data robustly into image files.
"""

import gzip
import json

# BCH error correction
import bchlib
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

correctable_bits = 16
block_length = 2019
code_block_len = 1920


def bit_shuffle(data_bytes, w, h):
    """
    Shuffle the bits of input data into a specific pattern based on image dimensions.

    :param data_bytes: Input data bytes to be shuffled
    :type data_bytes: bytes
    :param w: Width of the image
    :type w: int
    :param h: Height of the image
    :type h: int
    :return: Tuple containing shuffled data, dimension, rest tile size, and rest dimension
    :rtype: tuple(bytearray, int, int, int)
    """
    bits = np.frombuffer(data_bytes, dtype=np.uint8)
    bit_fac = 1
    bits = bits.reshape((h, w, 3 * bit_fac))
    flat_tile_len = (w * h * 3) // code_block_len
    tile_w = 32
    if flat_tile_len // tile_w > 100:
        tile_w = 64
    tile_h = flat_tile_len // tile_w
    h_cutoff = (h // tile_h) * tile_h
    tile_hr = h - h_cutoff
    easy_tiles = bits[:h_cutoff].reshape(h_cutoff // tile_h, tile_h, w // tile_w, tile_w, 3 * bit_fac)
    easy_tiles = easy_tiles.swapaxes(1, 2)
    easy_tiles = easy_tiles.reshape(-1, tile_h * tile_w)
    easy_tiles = easy_tiles.T
    rest_tiles = bits[h_cutoff:]
    rest_tiles = rest_tiles.reshape(tile_hr, 1, w // tile_w, tile_w, 3 * bit_fac)
    rest_tiles = rest_tiles.swapaxes(1, 2)
    rest_tiles = rest_tiles.reshape(-1, tile_hr * tile_w)
    rest_tiles = rest_tiles.T
    rest_dim = rest_tiles.shape[-1]
    rest_tiles = np.pad(rest_tiles, ((0, 0), (0, easy_tiles.shape[-1] - rest_tiles.shape[-1])), mode='constant',
                        constant_values=0)
    bits = np.concatenate((easy_tiles, rest_tiles), axis=0)
    dim = bits.shape[-1]
    bits = bits.reshape((-1,))
    return bytearray(bits.tobytes()), dim, rest_tiles.shape[0], rest_dim


def split_byte_ranges(data_bytes, n, w, h):
    """
    Split the input data bytes into chunks after shuffling.

    :param data_bytes: Input data bytes
    :type data_bytes: bytes
    :param n: Size of each chunk
    :type n: int
    :param w: Width of the image
    :type w: int
    :param h: Height of the image
    :type h: int
    :return: Tuple containing list of chunks, dimension, rest size, and rest dimension
    :rtype: tuple(list, int, int, int)
    """
    # noinspection PyUnresolvedReferences
    data_bytes, dim, rest_size, rest_dim = bit_shuffle(data_bytes.copy(), w, h)
    chunks = []
    for i in range(0, len(data_bytes), n):
        chunks.append(data_bytes[i:i + n])
    return chunks, dim, rest_size, rest_dim


def pad(data_bytes):
    """
    Pad the input data bytes to a fixed length of 2019 bytes.

    :param data_bytes: Input data bytes
    :type data_bytes: bytes
    :return: Padded data bytes
    :rtype: bytearray
    """
    return bytearray(data_bytes + b'\x00' * (2019 - len(data_bytes)))


def fec_encode(data_bytes, w, h):
    """
    Perform Forward Error Correction (FEC) encoding on the input data.

    :param data_bytes: Input data bytes
    :type data_bytes: bytes
    :param w: Width of the image
    :type w: int
    :param h: Height of the image
    :type h: int
    :return: FEC encoded data
    :rtype: bytes
    """
    # noinspection PyArgumentList
    encoder = bchlib.BCH(16, prim_poly=17475)
    chunks = [bytearray(encoder.encode(pad(x))) for x in split_byte_ranges(data_bytes, 2019, w, h)[0]]
    return b''.join(chunks)


class LSBInjector:
    """
    A class for injecting data into the least significant bits of image pixels.
    """

    def __init__(self, data):
        """
        Initialize the LSBInjector with image data.

        :param data: Image data
        :type data: numpy.ndarray
        """
        self.data = data
        self.buffer = bytearray()

    def put_32bit_integer(self, integer_value):
        """
        Add a 32-bit integer to the buffer.

        :param integer_value: Integer to be added
        :type integer_value: int
        """
        self.buffer.extend(integer_value.to_bytes(4, byteorder='big'))

    def put_bytes(self, bytes_list):
        """
        Add bytes to the buffer.

        :param bytes_list: Bytes to be added
        :type bytes_list: bytes
        """
        self.buffer.extend(bytes_list)

    def put_string(self, string):
        """
        Add a string to the buffer (encoded as UTF-8).

        :param string: String to be added
        :type string: str
        """
        self.put_bytes(string.encode('utf-8'))

    def finalize(self):
        """
        Finalize the injection process by embedding the buffer data into the image's least significant bits.
        """
        buffer = np.frombuffer(self.buffer, dtype=np.uint8)
        buffer = np.unpackbits(buffer)
        data = self.data[..., -1].T
        h, w = data.shape
        data = data.reshape((-1,))
        data[:] = 0xff
        buf_len = buffer.shape[0]
        data[:buf_len] = 0xfe
        data[:buf_len] = np.bitwise_or(data[:buf_len], buffer)
        data = data.reshape((h, w)).T
        self.data[..., -1] = data


def serialize_metadata(metadata: PngInfo) -> bytes:
    """
    Serialize PNG metadata into a compressed byte string.

    :param metadata: PNG metadata
    :type metadata: PIL.PngImagePlugin.PngInfo
    :return: Serialized and compressed metadata
    :rtype: bytes
    """
    data = {
        k: v
        for k, v in [
            data[1]
            .decode("latin-1" if data[0] == b"tEXt" else "utf-8")
            .split("\x00" if data[0] == b"tEXt" else "\x00\x00\x00\x00\x00")
            for data in metadata.chunks
            if data[0] == b"tEXt" or data[0] == b"iTXt"
        ]
    }
    data_encoded = json.dumps(data)
    return gzip.compress(bytes(data_encoded, "utf-8"))


def inject_data(image: Image.Image, data: PngInfo) -> Image.Image:
    """
    Inject metadata and error correction data into an image.

    :param image: Input image
    :type image: PIL.Image.Image
    :param data: PNG metadata to be injected
    :type data: PIL.PngImagePlugin.PngInfo
    :return: Image with injected data
    :rtype: PIL.Image.Image
    """
    # noinspection PyTypeChecker
    rgb = np.array(image.convert('RGB'))
    image = image.convert('RGBA')
    w, h = image.size
    # noinspection PyTypeChecker
    pixels = np.array(image)
    injector = LSBInjector(pixels)
    injector.put_string("stealth_pngcomp")
    data = serialize_metadata(data)
    injector.put_32bit_integer(len(data) * 8)
    injector.put_bytes(data)
    fec_data = fec_encode(bytearray(rgb.tobytes()), w, h)
    injector.put_32bit_integer(len(fec_data) * 8)
    injector.put_bytes(fec_data)
    injector.finalize()
    return Image.fromarray(injector.data)
