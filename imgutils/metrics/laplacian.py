import cv2
import numpy as np

from ..data import load_image, ImageTyping

__all__ = [
    'laplacian_score'
]


def _variance_of_laplacian(d_image: np.ndarray):
    """
    Calculate the variance of Laplacian for a given image.

    :param d_image: The input image as a numpy array.
    :type d_image: np.ndarray
    :return: The variance of Laplacian.
    :rtype: float
    """
    return cv2.Laplacian(d_image, cv2.CV_64F).var()


def laplacian_score(image: ImageTyping) -> float:
    """
    Calculate the Laplacian score for the given image.

    The Laplacian score is a measure of image bluriness.

    :param image: The input image.
    :type image: ImageTyping
    :return: The Laplacian score.
    :rtype: float

    Examples::
        >>> from imgutils.metrics import laplacian_score
        >>>
        >>> laplacian_score('laplacian/hutao.jpg')
        156.68285005210006
        >>> laplacian_score('laplacian/text_blur.png')
        2276.66629157129
        >>> laplacian_score('laplacian/real2.png')
        15.908745781486806
        >>> laplacian_score('laplacian/mmd.png')
        1072.8372572065527
    """
    v = np.array(load_image(image, force_background='white', mode='L'))
    return float(_variance_of_laplacian(v).item())
