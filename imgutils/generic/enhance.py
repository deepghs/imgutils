import numpy as np
from PIL import Image

from ..data import ImageTyping, load_image

__all__ = [
    'ImageEnhancer',
]


def _has_alpha_channel(image: Image.Image) -> bool:
    """
    Check if the image has an alpha channel.

    :param image: The image to check.
    :type image: Image.Image

    :return: True if the image has an alpha channel, False otherwise.
    :rtype: bool
    """
    return any(band in {'A', 'a', 'P'} for band in image.getbands())


class ImageEnhancer:
    def _process_rgb(self, rgb_array: np.ndarray):
        # input: a (3, H, W) float32[0.0, 1.0] array
        # output: another (3, H', W') float32[0.0, 1.0] array
        raise NotImplementedError

    def _process_alpha_channel_with_model(self, alpha_array: np.ndarray):
        assert len(alpha_array.shape) == 2, f'Alpha array should be 2-dim, but {alpha_array.shape!r} found.'
        enhanced_alpha_array = self._process_rgb(np.stack([alpha_array, alpha_array, alpha_array])).mean(axis=0)
        return enhanced_alpha_array

    def _process_rgba(self, rgba_array: np.ndarray):
        assert len(rgba_array.shape) == 3 and rgba_array.shape[0] == 4, \
            f'RGBA array should be 3-dim and 4-channels, but {rgba_array.shape!r} found.'

        return np.concatenate([
            self._process_rgb(rgba_array[:3, ...]),
            self._process_alpha_channel_with_model(rgba_array[3, ...])[None, ...]
        ], axis=0)

    def process(self, image: ImageTyping):
        image = load_image(image, mode=None, force_background=None)
        mode = 'RGBA' if _has_alpha_channel(image) else 'RGB'
        image = load_image(image, mode=mode, force_background=None)
        input_array = (np.array(image).astype(np.float32) / 255.0).transpose((2, 0, 1))
        if _has_alpha_channel(image):
            output_array = self._process_rgba(input_array)
        else:
            output_array = self._process_rgb(input_array)
        output_array = (np.clip(output_array, a_min=0.0, a_max=1.0) * 255.0).astype(np.uint8).transpose((1, 2, 0))
        return Image.fromarray(output_array, mode=mode)
