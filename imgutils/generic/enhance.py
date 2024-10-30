"""
Overview:
    Generic tools for image enhancement models.
"""
import numpy as np
from PIL import Image

from ..data import ImageTyping, load_image, has_alpha_channel

__all__ = [
    'ImageEnhancer',
]


class ImageEnhancer:
    """
    Enhances images by applying various processing techniques.

    This class provides methods to enhance images, including processing RGB images,
    alpha channels, and RGBA images.

    Methods:
        process: Enhances the input image.

    Private Methods:
        _process_rgb: Processes the RGB channels of an image.
        _process_alpha_channel_with_model: Processes the alpha channel using a model.
        _process_rgba: Processes RGBA images.

    Attributes:
        None
    """

    def _process_rgb(self, rgb_array: np.ndarray):
        """
        Process the RGB channels of an image.

        This method should be implemented in subclasses.

        :param rgb_array: The RGB channels of the image as a numpy array.
        :type rgb_array: np.ndarray

        :return: The processed RGB channels.
        :rtype: np.ndarray

        :raises NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError  # pragma: no cover

    def _process_alpha_channel_with_model(self, alpha_array: np.ndarray):
        """
        Process the alpha channel using a model.

        :param alpha_array: The alpha channel of the image as a numpy array.
        :type alpha_array: np.ndarray

        :return: The processed alpha channel.
        :rtype: np.ndarray
        """
        assert len(alpha_array.shape) == 2, f'Alpha array should be 2-dim, but {alpha_array.shape!r} found.'
        enhanced_alpha_array = self._process_rgb(np.stack([alpha_array, alpha_array, alpha_array])).mean(axis=0)
        return enhanced_alpha_array

    def _process_rgba(self, rgba_array: np.ndarray):
        """
        Process RGBA images.

        :param rgba_array: The RGBA image as a numpy array.
        :type rgba_array: np.ndarray

        :return: The processed RGBA image.
        :rtype: np.ndarray
        """
        assert len(rgba_array.shape) == 3 and rgba_array.shape[0] == 4, \
            f'RGBA array should be 3-dim and 4-channels, but {rgba_array.shape!r} found.'

        return np.concatenate([
            self._process_rgb(rgba_array[:3, ...]),
            self._process_alpha_channel_with_model(rgba_array[3, ...])[None, ...]
        ], axis=0)

    def process(self, image: ImageTyping):
        """
        Enhances the input image.

        :param image: The input image.
        :type image: ImageTyping

        :return: The enhanced image.
        :rtype: Image.Image
        """
        image = load_image(image, mode=None, force_background=None)
        mode = 'RGBA' if has_alpha_channel(image) else 'RGB'
        image = load_image(image, mode=mode, force_background=None)
        input_array = (np.array(image).astype(np.float32) / 255.0).transpose((2, 0, 1))
        if has_alpha_channel(image):
            output_array = self._process_rgba(input_array)
        else:
            output_array = self._process_rgb(input_array)
        output_array = (np.clip(output_array, a_min=0.0, a_max=1.0) * 255.0).astype(np.uint8).transpose((1, 2, 0))
        return Image.fromarray(output_array, mode=mode)
