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
    try:
        Image.open(path).load()
    except OSError:
        return True
    else:
        return False
