from contextlib import contextmanager
from threading import Lock

from PIL import Image, ImageFile

__all__ = [
    'is_truncated_file',
]

_LOCK = Lock()


@contextmanager
def _mock_load_truncated_images(value: bool):
    with _LOCK:
        _load = ImageFile.LOAD_TRUNCATED_IMAGES
        ImageFile.LOAD_TRUNCATED_IMAGES = value
        try:
            yield
        finally:
            ImageFile.LOAD_TRUNCATED_IMAGES = _load


@_mock_load_truncated_images(False)
def is_truncated_file(path: str) -> bool:
    """
    Overview:
        Check if an image is truncated or not.

    :param path: Path of file (must be a string which represents the path of image).
    :return: Is truncated or not.

    Examples:
        Here are some images for example

        .. image:: truncated.plot.py.svg
           :align: center

        >>> from imgutils.validate import is_truncated_file
        >>>
        >>> is_truncated_file('jpeg_full.jpeg')
        False
        >>> is_truncated_file('jpeg_truncated.jpeg')
        True
        >>> is_truncated_file('6125785.png')
        False
        >>> is_truncated_file('2216614_truncated.jpg')
        True

    .. warning::
        The function :func:`is_truncated_file` is thread-safe due to the usage of a global lock. \
        During the function is run, the value of `ImageFile.LOAD_TRUNCATED_IMAGES <https://pillow.readthedocs.io/en/stable/reference/ImageFile.html#PIL.ImageFile.LOAD_TRUNCATED_IMAGES>`_ \
        is set to ``True``, this may cause some side effect when your projects have dependency on this ``ImageFile.LOAD_TRUNCATED_IMAGES``.

    """
    try:
        Image.open(path).load()
    except OSError:
        return True
    else:
        return False
