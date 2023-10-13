from functools import lru_cache
from typing import Tuple, Any

import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

from ..data import ImageTyping, load_image
from ..utils import open_onnx_model, area_batch_run


@lru_cache()
def _open_cdc_upscaler_model(model: str) -> Tuple[Any, int]:
    ort = open_onnx_model(hf_hub_download(
        f'deepghs/cdc_anime_onnx',
        f'{model}.onnx'
    ))

    input_ = np.random.randn(1, 3, 16, 16).astype(np.float32)
    output_, = ort.run(['output'], {'input': input_})

    batch, channels, scale_h, height, scale_w, width = output_.shape
    assert batch == 1 and channels == 3 and height == 16 and width == 16, \
        f'Unexpected output size found {output_.shape!r}.'
    assert scale_h == scale_w, f'Scale of height and width not match - {output_.shape!r}.'

    return ort, scale_h


def upscale_with_cdc(image: ImageTyping, model: str = 'HGSR-MHR-anime-aug_X4_320',
                     tile_size: int = 512, tile_overlap: int = 64, batch_size: int = 1,
                     silent: bool = False) -> Image.Image:
    image = load_image(image, mode='RGB', force_background='white')
    input_ = np.array(image).astype(np.float32) / 255.0
    input_ = input_.transpose((2, 0, 1))[None, ...]

    ort, scale = _open_cdc_upscaler_model(model)

    def _method(ix):
        ox, = ort.run(['output'], {'input': ix.astype(np.float32)})
        batch, channels, scale_, height, scale_, width = ox.shape
        return ox.reshape((batch, channels, scale_ * height, scale_ * width))

    output_ = area_batch_run(
        input_, _method,
        tile_size=tile_size, tile_overlap=tile_overlap, batch_size=batch_size,
        scale=scale, silent=silent, process_title='CDC Upscale',
    )
    output_ = np.clip(output_, a_min=0.0, a_max=1.0)
    return Image.fromarray((output_[0].transpose((1, 2, 0)) * 255).astype(np.int8), 'RGB')
