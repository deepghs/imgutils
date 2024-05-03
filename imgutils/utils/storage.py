import os
from functools import lru_cache

__all__ = [
    'get_storage_dir',
]


@lru_cache()
def get_storage_dir():
    """
    Get the storage directory path for image utilities.

    :return: The path to the storage directory.
    :rtype: str
    """
    dir_ = os.path.abspath(
        os.environ.get('IU_HOME') or os.path.expanduser(os.path.join('~', '.cache', 'dghs-imgutils')))
    os.makedirs(dir_, exist_ok=True)
    return dir_
