import os.path
import re
from functools import lru_cache
from typing import Optional, Mapping, Tuple

import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download, HfFileSystem

from ..data import ImageTyping, load_image
from ..utils import open_onnx_model, area_batch_run

_hf_fs = HfFileSystem()

_REPOSITORY = 'deepghs/waifu2x_onnx'
_FILENAME_PATTERN = re.compile(r'^(noise(?P<noise>\d+)_?)?(scale(?P<scale>\d+)x)?\.onnx$')


@lru_cache()
def _load_available_version() -> Mapping[Tuple[str, str, str], Mapping[Tuple[Optional[int], int], str]]:
    records = {}
    for file in _hf_fs.glob(f'{_REPOSITORY}/*/onnx_models/*/*/*.onnx'):
        segments = os.path.relpath(file, _REPOSITORY).split('/')
        assert len(segments) == 5 and segments[1] == 'onnx_models'
        version, model, type_ = segments[0], segments[2], segments[3]
        filename = segments[4]

        key = (version, model, type_)
        if key not in records:
            records[key] = []
        records[key].append(filename)

    retval = {}
    for key, filenames in records.items():
        retval[key] = {}
        for filename in filenames:
            matching = _FILENAME_PATTERN.fullmatch(filename)
            assert matching, f'Not matched, {filename!r}, key: {key!r}'
            noise = int(matching.group('noise')) if matching.group('noise') else None
            scale = int(matching.group('scale')) if matching.group('scale') else 1
            retval[key][(noise, scale)] = filename

    return retval


@lru_cache()
def _open_waifu2x_onnx_model(version: str, model: str, type_: str, noise: Optional[int], scale: int):
    _all_versions = _load_available_version()
    if (version, model, type_) in _all_versions:
        _all_k = _all_versions[(version, model, type_)]
        if (noise, scale) in _all_k:
            filename = _all_k[(noise, scale)]
            return open_onnx_model(hf_hub_download(
                f'deepghs/waifu2x_onnx',
                f'{version}/onnx_models/{model}/{type_}/{filename}',
            ))
        else:
            raise ValueError(f'Noise {noise!r} or scale {scale!r} may be invalid.')
    else:
        raise ValueError(f"Version {version!r} or model {model!r} or type_ {type_!r} may be invalid.")


def _single_upscale_by_waifu2x(x, version: str = '20230504', model: str = 'swin_unet',
                               type_: str = 'art', noise: Optional[int] = None, scale: int = 2):
    ort = _open_waifu2x_onnx_model(version, model, type_, noise, scale)
    # noinspection PyTypeChecker
    x = np.pad(x, ((0, 0), (0, 0), (8, 8), (8, 8)), mode='reflect')
    y, = ort.run(['y'], {'x': x})
    return y


def upscale_image_by_waifu2x(image: ImageTyping, scale: int = 2, noise: Optional[int] = None,
                             version: str = '20230504', model: str = 'swin_unet', type_: str = 'art',
                             tile_size: int = 64, tile_overlap: int = 8, silent: bool = False) -> Image.Image:
    image = load_image(image, mode='RGB', force_background='white')
    input_ = np.array(image).astype(np.float32) / 255.0
    input_ = input_.transpose((2, 0, 1))[None, ...]

    def _method(ix):
        return _single_upscale_by_waifu2x(ix, version, model, type_, noise, scale)

    output_ = area_batch_run(
        input_, _method,
        scale=scale, tile_size=tile_size, tile_overlap=tile_overlap, silent=silent,
        process_title='Waifu2x Upscale',
    )
    output_ = np.clip(output_, a_min=0.0, a_max=1.0)
    return Image.fromarray((output_[0].transpose((1, 2, 0)) * 255).astype(np.int8), 'RGB')
