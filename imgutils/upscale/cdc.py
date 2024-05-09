from functools import lru_cache
from typing import Tuple, Any

import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

from .transparent import _rgba_preprocess, _rgba_postprocess
from ..data import ImageTyping, load_image
from ..utils import open_onnx_model, area_batch_run


@lru_cache()
def _open_cdc_upscaler_model(model: str) -> Tuple[Any, int]:
    """
    Opens and initializes the CDC upscaler model.

    :param model: The name of the model to use.
    :type model: str

    :return: Tuple of the ONNX model and the scale factor.
    :rtype: Tuple[Any, int]
    """
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


_CDC_INPUT_UNIT = 16


def upscale_with_cdc(image: ImageTyping, model: str = 'HGSR-MHR-anime-aug_X4_320',
                     tile_size: int = 512, tile_overlap: int = 64, batch_size: int = 1,
                     silent: bool = False) -> Image.Image:
    """
    Upscale the input image using the CDC upscaler model.

    :param image: The input image.
    :type image: ImageTyping

    :param model: The name of the model to use. (default: 'HGSR-MHR-anime-aug_X4_320')
    :type model: str

    :param tile_size: The size of each tile. (default: 512)
    :type tile_size: int

    :param tile_overlap: The overlap between tiles. (default: 64)
    :type tile_overlap: int

    :param batch_size: The batch size. (default: 1)
    :type batch_size: int

    :param silent: Whether to suppress progress messages. (default: False)
    :type silent: bool

    :return: The upscaled image.
    :rtype: Image.Image
    """
    image, alpha_mask = _rgba_preprocess(image)
    image = load_image(image, mode='RGB', force_background='white')
    input_ = np.array(image).astype(np.float32) / 255.0
    input_ = input_.transpose((2, 0, 1))[None, ...]

    ort, scale = _open_cdc_upscaler_model(model)

    def _method(ix):
        ix = ix.astype(np.float32)
        batch, channels, height, width = ix.shape
        p_height = 0 if height % _CDC_INPUT_UNIT == 0 else _CDC_INPUT_UNIT - (height % _CDC_INPUT_UNIT)
        p_width = 0 if width % _CDC_INPUT_UNIT == 0 else _CDC_INPUT_UNIT - (width % _CDC_INPUT_UNIT)
        if p_height > 0 or p_width > 0:  # align to 16
            ix = np.pad(ix, ((0, 0), (0, 0), (0, p_height), (0, p_width)), mode='reflect')
        actual_height, actual_width = height, width

        ox, = ort.run(['output'], {'input': ix})
        batch, channels, scale_, height, scale_, width = ox.shape
        ox = ox.reshape((batch, channels, scale_ * height, scale_ * width))
        ox = ox[..., :scale_ * actual_height, :scale_ * actual_width]  # crop back
        return ox

    output_ = area_batch_run(
        input_, _method,
        tile_size=tile_size, tile_overlap=tile_overlap, batch_size=batch_size,
        scale=scale, silent=silent, process_title='CDC Upscale',
    )
    output_ = np.clip(output_, a_min=0.0, a_max=1.0)
    ret_image = Image.fromarray((output_[0].transpose((1, 2, 0)) * 255).astype(np.uint8), 'RGB')
    return _rgba_postprocess(ret_image, alpha_mask)
