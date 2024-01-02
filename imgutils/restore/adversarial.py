"""
Overview:
    Useful tools to remove adversarial noises, just using opencv library without any models.

    .. image:: adversarial_demo.plot.py.svg
        :align: center

    This is an overall benchmark of all the adversarial denoising:

    .. image:: adversarial_benchmark.plot.py.svg
        :align: center

    .. note::
        This tool is inspired from `Huggingface - mf666/mist-fucker <https://huggingface.co/spaces/mf666/mist-fucker>`_.
"""
import random

import cv2
import numpy as np
from PIL import Image
from cv2.ximgproc import guidedFilter

from ..data import load_image


def remove_adversarial_noise(
        image: Image.Image, diameter_min: int = 4, diameter_max: int = 6,
        sigma_color_min: float = 6.0, sigma_color_max: float = 10.0,
        sigma_space_min: float = 6.0, sigma_space_max: float = 10.0,
        radius_min: int = 3, radius_max: int = 6, eps_min: float = 16.0, eps_max: float = 24.0,
        b_iters: int = 64, g_iters: int = 8,
) -> Image.Image:
    """
    Remove adversarial noise from an image using random bilateral and guided filtering.

    This function applies random bilateral filtering and random guided filtering iteratively to reduce adversarial noise
    in the input image.

    :param image: The input image.
    :type image: Image.Image

    :param diameter_min: Minimum diameter for bilateral filtering.
    :type diameter_min: int, optional

    :param diameter_max: Maximum diameter for bilateral filtering.
    :type diameter_max: int, optional

    :param sigma_color_min: Minimum filter sigma in the color space for bilateral filtering.
    :type sigma_color_min: float, optional

    :param sigma_color_max: Maximum filter sigma in the color space for bilateral filtering.
    :type sigma_color_max: float, optional

    :param sigma_space_min: Minimum filter sigma in the coordinate space for bilateral filtering.
    :type sigma_space_min: float, optional

    :param sigma_space_max: Maximum filter sigma in the coordinate space for bilateral filtering.
    :type sigma_space_max: float, optional

    :param radius_min: Minimum radius for guided filtering.
    :type radius_min: int, optional

    :param radius_max: Maximum radius for guided filtering.
    :type radius_max: int, optional

    :param eps_min: Minimum guided filter regularization term.
    :type eps_min: float, optional

    :param eps_max: Maximum guided filter regularization term.
    :type eps_max: float, optional

    :param b_iters: Number of iterations for bilateral filtering.
    :type b_iters: int, optional

    :param g_iters: Number of iterations for guided filtering.
    :type g_iters: int, optional

    :return: Image with adversarial noise removed.
    :rtype: Image.Image
    """
    image = load_image(image, mode='RGB', force_background='white')
    img = np.array(image).astype(np.float32)
    y = img.copy()

    # Apply random bilateral filtering iteratively
    for _ in range(b_iters):
        diameter = random.randint(diameter_min, diameter_max)
        sigma_color = random.uniform(sigma_color_min, sigma_color_max)
        sigma_space = random.uniform(sigma_space_min, sigma_space_max)
        y = cv2.bilateralFilter(y, diameter, sigma_color, sigma_space)

    # Apply random guided filtering iteratively
    for _ in range(g_iters):
        radius = random.randint(radius_min, radius_max)
        eps = random.uniform(eps_min, eps_max)
        y = guidedFilter(img, y, radius, eps)

    # Clip the values and convert back to uint8 for PIL Image
    output_image = Image.fromarray(y.clip(0, 255).astype(np.uint8))
    return output_image
