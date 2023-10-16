"""
Overview:
    Restore the images using `NafNet <https://github.com/megvii-research/NAFNet>`_.

    .. image:: nafnet_demo.plot.py.svg
        :align: center

    This is an overall benchmark of all the NafNet models:

    .. image:: nafnet_benchmark.plot.py.svg
        :align: center

    .. warning::
        Currently, we've identified a significant issue with NafNet when images contain gaussian noise.
        To ensure your code functions correctly, please ensure the credibility of
        your image source or preprocess them using SCUNet.
"""
from functools import lru_cache

import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

from ..data import ImageTyping, load_image
from ..utils import open_onnx_model, area_batch_run

try:
    from typing import Literal
except (ImportError, ModuleNotFoundError):
    from typing_extensions import Literal

NafNetModelTyping = Literal['REDS', 'GoPro', 'SIDD']


@lru_cache()
def _open_nafnet_model(model: NafNetModelTyping):
    """
    Open the NAFNet model for image restoration.

    :param model: The NAFNet model type ('REDS', 'GoPro', 'SIDD').
    :type model: NafNetModelTyping
    :return: The NAFNet ONNX model.
    """
    return open_onnx_model(hf_hub_download(
        f'deepghs/image_restoration',
        f'NAFNet-{model}-width64.onnx',
    ))


def restore_with_nafnet(image: ImageTyping, model: NafNetModelTyping = 'REDS',
                        tile_size: int = 256, tile_overlap: int = 16, silent: bool = False) -> Image.Image:
    """
    Restore an image using the NAFNet model.

    :param image: The input image.
    :type image: ImageTyping
    :param model: The NAFNet model type ('REDS', 'GoPro', 'SIDD'). Default is 'REDS'.
    :type model: NafNetModelTyping
    :param tile_size: The size of processing tiles. Default is 256.
    :type tile_size: int
    :param tile_overlap: The overlap between tiles. Default is 16.
    :type tile_overlap: int
    :param silent: If True, the progress will not be displayed. Default is False.
    :type silent: bool
    :return: The restored image.
    :rtype: Image.Image
    """
    image = load_image(image, mode='RGB', force_background='white')
    input_ = np.array(image).astype(np.float32) / 255.0
    input_ = input_.transpose((2, 0, 1))[None, ...]

    def _method(ix):
        ox, = _open_nafnet_model(model).run(['output'], {'input': ix})
        return ox

    output_ = area_batch_run(
        input_, _method,
        tile_size=tile_size, tile_overlap=tile_overlap, silent=silent,
        process_title='NafNet Restore',
    )
    output_ = np.clip(output_, a_min=0.0, a_max=1.0)
    return Image.fromarray((output_[0].transpose((1, 2, 0)) * 255).astype(np.int8), 'RGB')
