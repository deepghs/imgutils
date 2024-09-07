import gzip
import json

import numpy as np
from PIL import Image


# MIT: https://github.com/NovelAI/novelai-image-metadata/blob/main/nai_meta.py
class LSBExtractor(object):
    def __init__(self, data: np.ndarray):
        self.data = data
        self.rows, self.cols, self.dim = data.shape
        self.bits = 0
        self.byte = 0
        self.row = 0
        self.col = 0

    def _extract_next_bit(self):
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
        while self.bits < 8:
            self._extract_next_bit()
        byte = bytearray([self.byte])
        self.bits = 0
        self.byte = 0
        return byte

    def get_next_n_bytes(self, n):
        bytes_list = bytearray()
        for _ in range(n):
            byte = self.get_one_byte()
            if not byte:
                break
            bytes_list.extend(byte)
        return bytes_list

    def read_32bit_integer(self):
        bytes_list = self.get_next_n_bytes(4)
        if len(bytes_list) == 4:
            integer_value = int.from_bytes(bytes_list, byteorder='big')
            return integer_value
        else:
            return None


# MIT: https://github.com/NovelAI/novelai-image-metadata/blob/main/nai_meta.py
class ImageLsbDataExtractor(object):
    def __init__(self, magic: str = "stealth_pngcomp"):
        self._magic_bytes = magic.encode('utf-8')

    def extract_data(self, image: Image.Image) -> dict:
        if image.mode != 'RGBA':
            raise ValueError(f'Image should be in RGBA mode, but {image.mode!r} found.')
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
