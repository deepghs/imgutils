"""
Overview:
    A utility for aligning dimensions based on the size of an image.
"""
from PIL import Image

from imgutils.data import ImageTyping, load_image


def align_maxsize(image: ImageTyping, max_size: int) -> Image.Image:
    """
    Resizes the image while maintaining its aspect ratio, ensuring that the length of its longer side
    aligns with the given ``max_size``.

    :param image: The input image to be resized.
    :param max_size: The maximum length of the longer side after resizing.
    :return: The resized image.

    Example::
        >>> from PIL import Image
        >>> from imgutils.operate import align_maxsize
        >>>
        >>> image = Image.open('genshin_post.jpg')
        >>> image.size
        (1280, 720)
        >>>
        >>> new_img = align_maxsize(image, max_size=600)
        >>> new_img.size
        (600, 337)
    """
    image = load_image(image, force_background=None)
    width, height = image.size

    r = max_size / max(width, height)
    new_width, new_height = int(width * r), int(height * r)
    return image.resize((new_width, new_height))
