"""
Overview:
    Restore the images using `SCUNet <https://github.com/cszn/SCUNet>`_.

    .. image:: scunet_demo.plot.py.svg
        :align: center

    This is an overall benchmark of all the SCUNet models:

    .. image:: scunet_benchmark.plot.py.svg
        :align: center

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

SCUNetModelTyping = Literal['GAN', 'PSNR']


@lru_cache()
def _open_scunet_model(model: SCUNetModelTyping):
    """
    Open the SCUNet model for image restoration.

    :param model: The SCUNet model type ('GAN', 'PSNR').
    :type model: SCUNetModelTyping
    :return: The SCUNet ONNX model.
    """
    return open_onnx_model(hf_hub_download(
        f'deepghs/image_restoration',
        f'SCUNet-{model}.onnx'
    ))


def restore_with_scunet(image: ImageTyping, model: SCUNetModelTyping = 'GAN',
                        tile_size: int = 128, tile_overlap: int = 16, silent: bool = False) -> Image.Image:
    """
    Restore an image using the SCUNet model.

    :param image: The input image.
    :type image: ImageTyping
    :param model: The SCUNet model type ('GAN', 'PSNR'). Default is 'GAN'.
    :type model: SCUNetModelTyping
    :param tile_size: The size of processing tiles. Default is 128.
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
        ox, = _open_scunet_model(model).run(['output'], {'input': ix})
        return ox

    output_ = area_batch_run(
        input_, _method,
        tile_size=tile_size, tile_overlap=tile_overlap, silent=silent,
        process_title='SCUNet Restore',
    )
    output_ = np.clip(output_, a_min=0.0, a_max=1.0)
    return Image.fromarray((output_[0].transpose((1, 2, 0)) * 255).astype(np.int8), 'RGB')
