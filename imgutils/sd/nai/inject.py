# MIT: https://github.com/NovelAI/novelai-image-metadata/blob/main/nai_meta_writer.py
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
    data_bytes, dim, rest_size, rest_dim = bit_shuffle(data_bytes.copy(), w, h)
    chunks = []
    for i in range(0, len(data_bytes), n):
        chunks.append(data_bytes[i:i + n])
    return chunks, dim, rest_size, rest_dim


def pad(data_bytes):
    return bytearray(data_bytes + b'\x00' * (2019 - len(data_bytes)))


# Returns codes for the data in data_bytes
def fec_encode(data_bytes, w, h):
    # noinspection PyArgumentList
    encoder = bchlib.BCH(16, prim_poly=17475)
    # import galois
    # encoder = galois.BCH(16383, 16383-224, d=17, c=224)
    chunks = [bytearray(encoder.encode(pad(x))) for x in split_byte_ranges(data_bytes, 2019, w, h)[0]]
    return b''.join(chunks)


class LSBInjector:
    def __init__(self, data):
        self.data = data
        self.buffer = bytearray()

    def put_32bit_integer(self, integer_value):
        self.buffer.extend(integer_value.to_bytes(4, byteorder='big'))

    def put_bytes(self, bytes_list):
        self.buffer.extend(bytes_list)

    def put_string(self, string):
        self.put_bytes(string.encode('utf-8'))

    def finalize(self):
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
    # Extract metadata from PNG chunks
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
    # Encode and compress data using gzip
    data_encoded = json.dumps(data)
    return gzip.compress(bytes(data_encoded, "utf-8"))


def inject_data(image: Image.Image, data: PngInfo) -> Image.Image:
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
