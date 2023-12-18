"""
Overview:
    Useful tools to remove adversarial noises, just using opencv library without any models.

    .. image:: adversarial_demo.plot.py.svg
        :align: center

    This is an overall benchmark of all the adversarial denoising:

    .. image:: adversarial_benchmark.plot.py.svg
        :align: center

    .. note::
        This tool is inspired from `Github - lllyasviel/AdverseCleaner <https://github.com/lllyasviel/AdverseCleaner>`_.
"""
import cv2
import numpy as np
from PIL import Image
from cv2.ximgproc import guidedFilter

from ..data import load_image, ImageTyping


def remove_adversarial_noise(image: ImageTyping, diameter: int = 5, sigma_color: float = 8.0,
                             sigma_space: float = 8.0, radius: int = 4, eps: float = 16.0) -> Image.Image:
    """
    Remove adversarial noise from an image using bilateral and guided filtering.

    This function applies bilateral filtering and guided filtering to reduce adversarial noise in the input image.

    :param image: The input image.
    :type image: ImageTyping

    :param diameter: Diameter of each pixel neighborhood for bilateral filtering.
    :type diameter: int, optional

    :param sigma_color: Filter sigma in the color space for bilateral filtering.
    :type sigma_color: float, optional

    :param sigma_space: Filter sigma in the coordinate space for bilateral filtering.
    :type sigma_space: float, optional

    :param radius: Radius of Guided Filter.
    :type radius: float, optional

    :param eps: Guided Filter regularization term.
    :type eps: int, optional

    :return: Image with adversarial noise removed.
    :rtype: Image.Image
    """
    image = load_image(image, mode='RGB', force_background='white')
    img = np.array(image).astype(np.float32)
    y = img.copy()

    # Apply bilateral filtering iteratively
    for _ in range(64):
        y = cv2.bilateralFilter(y, diameter, sigma_color, sigma_space)

    # Apply guided filtering iteratively
    for _ in range(4):
        y = guidedFilter(img, y, radius, eps)

    # Clip the values and convert back to uint8 for PIL Image
    output_image = Image.fromarray(y.clip(0, 255).astype(np.uint8))
    return output_image
