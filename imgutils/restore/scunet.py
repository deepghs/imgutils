"""
Overview:
    Restore the images using `SCUNet <https://github.com/cszn/SCUNet>`_.

    .. image:: scunet_demo.plot.py.svg
        :align: center

    This is an overall benchmark of all the SCUNet models:

    .. image:: scunet_benchmark.plot.py.svg
        :align: center

    .. note::
        New in version v0.4.4, **images with alpha channel supported**.

        If you use an image with alpha channel (e.g. RGBA images),
        it will return a RGBA image, otherwise return RGG image.
"""
from typing import Literal

import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

from ..data import ImageTyping
from ..generic import ImageEnhancer
from ..utils import open_onnx_model, area_batch_run, ts_lru_cache

SCUNetModelTyping = Literal['GAN', 'PSNR']


@ts_lru_cache()
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


class _Enhancer(ImageEnhancer):
    def __init__(self, model: SCUNetModelTyping = 'GAN', tile_size: int = 128, tile_overlap: int = 16,
                 batch_size: int = 4, silent: bool = False):
        self.model = model
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.batch_size = batch_size
        self.silent = silent

    def _process_rgb(self, rgb_array: np.ndarray):
        input_ = rgb_array[None, ...]

        def _method(ix):
            ox, = _open_scunet_model(self.model).run(['output'], {'input': ix})
            return ox

        output_ = area_batch_run(
            input_, _method,
            tile_size=self.tile_size, tile_overlap=self.tile_overlap, batch_size=self.batch_size,
            silent=self.silent, process_title='SCUNet Restore',
        )
        output_ = np.clip(output_, a_min=0.0, a_max=1.0)
        return output_[0]


@ts_lru_cache()
def _get_enhancer(model: SCUNetModelTyping = 'GAN', tile_size: int = 128, tile_overlap: int = 16,
                  batch_size: int = 4, silent: bool = False) -> _Enhancer:
    return _Enhancer(model, tile_size, tile_overlap, batch_size, silent)


def restore_with_scunet(image: ImageTyping, model: SCUNetModelTyping = 'GAN',
                        tile_size: int = 128, tile_overlap: int = 16, batch_size: int = 4,
                        silent: bool = False) -> Image.Image:
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
    :param batch_size: The batch size of inference. Default is 4.
    :type batch_size: int
    :param silent: If True, the progress will not be displayed. Default is False.
    :type silent: bool
    :return: The restored image.
    :rtype: Image.Image
    """
    return _get_enhancer(model, tile_size, tile_overlap, batch_size, silent).process(image)
