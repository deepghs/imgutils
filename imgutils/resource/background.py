"""
Overview:
    Get background images.

    These resources are hosted on `deepghs/anime-bg <https://huggingface.co/datasets/deepghs/anime-bg>`_,
    which is based on `skytnt/anime-segmentation <https://huggingface.co/datasets/skytnt/anime-segmentation>`_.

    .. image:: background_full.plot.py.svg
        :align: center
"""
import os.path
from functools import lru_cache
from typing import Optional, List

import pandas as pd
from PIL import Image
from filelock import FileLock
from hfutils.index import hf_tar_file_download
from huggingface_hub import hf_hub_download

from ..data import load_image
from ..utils import get_storage_dir, ts_lru_cache

__all__ = [
    'BackgroundImageSet',
    'list_bg_image_files',
    'get_bg_image_file',
    'get_bg_image',
    'random_bg_image_file',
    'random_bg_image',
]

_BG_REPO = 'deepghs/anime-bg'


@ts_lru_cache()
def _global_df() -> pd.DataFrame:
    """
    Load the global dataframe containing information about background images.

    :return: The global dataframe containing information about background images.
    :rtype: pd.DataFrame
    """
    return pd.read_csv(hf_hub_download(
        repo_id=_BG_REPO,
        repo_type='dataset',
        filename='images.csv'
    ))


@lru_cache()
def _bg_root_dir() -> str:
    """
    Get the root directory for storing background images.

    :return: The root directory for storing background images.
    :rtype: str
    """
    root = os.path.join(get_storage_dir(), 'bg')
    os.makedirs(root, exist_ok=True)
    return root


class BackgroundImageSet:
    def __init__(self, width: Optional[float] = None, height: Optional[float] = None,
                 strict_level: float = 1.5, min_selected: int = 5,
                 min_width: Optional[int] = None, min_height: Optional[int] = None,
                 min_resolution: Optional[int] = None):
        """
        Initialize a BackgroundImageSet instance.

        :param width: The desired width of background images. (default: None)
        :type width: Optional[float]

        :param height: The desired height of background images. (default: None)
        :type height: Optional[float]

        :param strict_level: The strictness level for selecting images. (default: 1.5)
        :type strict_level: float

        :param min_selected: The minimum number of images to consider for selection. (default: 5)
        :type min_selected: int

        :param min_width: The minimum width of background images to consider. (default: None)
        :type min_width: Optional[int]

        :param min_height: The minimum height of background images to consider. (default: None)
        :type min_height: Optional[int]

        :param min_resolution: The minimum resolution of background images to consider. (default: None)
        :type min_resolution: Optional[int]
        """
        df = _global_df().copy()
        if min_width:
            df = df[df['width'] >= min_width]
        if min_height:
            df = df[df['height'] >= min_height]
        if min_resolution:
            df = df[(df['height'] * df['width']) >= min_resolution ** 2]

        if width and height:
            r1, r2 = width / height, height / width
            df['dx'] = (df['width'] / df['height'] - r1).abs() + (df['height'] / df['width'] - r2).abs()
            df = df.sort_values(['dx'], ascending=[True])
            df['i'] = range(len(df))
            idx = df['dx'] < (df['dx'].mean() - df['dx'].std() * strict_level)
            idx = idx | (df['i'] < max(min_selected, 1))
            idx = idx | (df['dx'] <= df[idx]['dx'].max())
            df = df[idx]
        elif width and not height:
            df['dist'] = (df['width'] - width).abs()
            df = df.sort_values(['dist'], ascending=[True])
            df['i'] = range(len(df))
            idx = df['dist'] < (df['dist'].mean() - df['dist'].std() * strict_level)
            idx = idx | (df['i'] < max(min_selected, 1))
            idx = idx | (df['dist'] <= df[idx]['dist'].max())
            df = df[idx]
        elif not width and height:
            df['dist'] = (df['height'] - height).abs()
            df = df.sort_values(['dist'], ascending=[True])
            df['i'] = range(len(df))
            idx = df['dist'] < (df['dist'].mean() - df['dist'].std() * strict_level)
            idx = idx | (df['i'] < max(min_selected, 1))
            idx = idx | (df['dist'] <= df[idx]['dist'].max())
            df = df[idx]
        else:
            pass

        self.df = df
        if len(self.df) == 0:
            raise ValueError('No background images selected, please lower your settings.')
        self._map = {item['filename']: item for item in self.df.to_dict('records')}

    def list_image_files(self) -> List[str]:
        """
        Get a list of filenames of background images.

        :return: A list of filenames of background images.
        :rtype: List[str]
        """
        return self.df['filename'].tolist()

    def get_image_file(self, filename: str) -> str:
        """
        Get the local file path of a background image by filename.

        :param filename: The filename of the background image.
        :type filename: str

        :return: The local file path of the background image.
        :rtype: str
        """
        return self._load_local_image_file(filename)

    def get_image(self, filename: str) -> Image.Image:
        """
        Get the PIL Image object of a background image by filename.

        :param filename: The filename of the background image.
        :type filename: str

        :return: The PIL Image object of the background image.
        :rtype: Image.Image
        """
        return load_image(self.get_image_file(filename), mode='RGB')

    def _random_filename(self):
        return self.df['filename'].sample(n=1).tolist()[0]

    def random_image_file(self) -> str:
        """
        Get the filename of a randomly selected background image.

        :return: The filename of a randomly selected background image.
        :rtype: str
        """
        return self.get_image_file(self._random_filename())

    def random_image(self) -> Image.Image:
        """
        Get the PIL Image object of a randomly selected background image.

        :return: The PIL Image object of a randomly selected background image.
        :rtype: Image.Image
        """
        return self.get_image(self._random_filename())

    def _load_local_image_file(self, filename):
        filename = os.path.normcase(filename)
        if filename not in self._map:
            raise FileNotFoundError(f'Background file {filename!r} not found.')
        info = self._map[filename]

        lock_file = os.path.join(_bg_root_dir(), f'{filename}.lock')
        with FileLock(lock_file):
            image_file = os.path.join(_bg_root_dir(), filename)
            if not os.path.exists(image_file):
                hf_tar_file_download(
                    repo_id=_BG_REPO,
                    archive_in_repo=f"images/{info['archive']}",
                    file_in_archive=info['filename'],
                    local_file=image_file
                )

            return image_file


@ts_lru_cache()
def _get_default_set():
    """
    Get the default BackgroundImageSet instance.

    :return: The default BackgroundImageSet instance.
    :rtype: BackgroundImageSet
    """
    return BackgroundImageSet()


def list_bg_image_files() -> List[str]:
    """
    Get a list of filenames of background images.

    :return: A list of filenames of background images.
    :rtype: List[str]

    Examples::
        >>> from imgutils.resource import list_bg_image_files
        >>>
        >>> files = list_bg_image_files()
        >>> type(files)
        <class 'list'>
        >>> len(files)
        8057
        >>> files[:5]
        ['000000.jpg', '000001.jpg', '000002.jpg', '000003.jpg', '000004.jpg']
    """
    return _get_default_set().list_image_files()


def get_bg_image_file(filename: str) -> str:
    """
    Get the local file path of a background image by filename.

    :param filename: The filename of the background image.
    :type filename: str

    :return: The local file path of the background image.
    :rtype: str

    Examples::
        >>> from imgutils.resource import get_bg_image_file
        >>>
        >>> get_bg_image_file('000001.jpg')
        '/home/user/.cache/dghs-imgutils/bg/000001.jpg'
    """
    return _get_default_set().get_image_file(filename)


def get_bg_image(filename) -> Image.Image:
    """
    Get the PIL Image object of a background image by filename.

    :param filename: The filename of the background image.
    :type filename: str

    :return: The PIL Image object of the background image.
    :rtype: Image.Image

    Examples::
        >>> from imgutils.resource import get_bg_image
        >>>
        >>> get_bg_image('000001.jpg')
        <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=2400x1600 at 0x7FEB86ED5160>
    """
    return _get_default_set().get_image(filename)


def random_bg_image_file() -> str:
    """
    Get the filename of a randomly selected background image.

    :return: The filename of a randomly selected background image.
    :rtype: str

    Examples::
        >>> from imgutils.resource import random_bg_image_file
        >>>
        >>> random_bg_image_file()
        '/home/user/.cache/dghs-imgutils/bg/003258.jpg'
    """
    return _get_default_set().random_image_file()


def random_bg_image() -> Image.Image:
    """
    Get the PIL Image object of a randomly selected background image.

    :return: The PIL Image object of a randomly selected background image.
    :rtype: Image.Image

    Examples::
        >>> from imgutils.resource import random_bg_image
        >>>
        >>> random_bg_image()
        <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=400x400 at 0x7FEB86A748B0>
    """
    return _get_default_set().random_image()
