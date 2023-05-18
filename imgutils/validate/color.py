from ..data import load_image, ImageTyping

__all__ = [
    'is_greyscale',
]

_GREYSCALE_K, _GREYSCALE_B = (-0.3578160565562811, 10.867076188331309)
_G_PNSR_THRESHOLD = -_GREYSCALE_B / _GREYSCALE_K


def is_greyscale(image: ImageTyping):
    """
    Overview:
    Check if an image is greyscale or not.

    :param image: Path or PIL object of image.
    :return: Is greyscale or not.

    Examples:
        Here are some images for example

        .. image:: greyscale.plot.py.svg
           :align: center

        >>> from imgutils.validate import is_greyscale
        >>>
        >>> is_greyscale('jpeg_full.jpeg')
        False
        >>> is_greyscale('6125901.jpg')
        False
        >>> is_greyscale('6125785.png')
        False
        >>> is_greyscale('6124220.jpg')
        True
    """
    from ..metrics import psnr

    image = load_image(image)
    return psnr(image, image.convert('L')) >= _G_PNSR_THRESHOLD
