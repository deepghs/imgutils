import os.path
from functools import lru_cache
from typing import Optional, List

import pandas as pd
from PIL import Image
from filelock import FileLock
from hfutils.index import hf_tar_file_download
from huggingface_hub import hf_hub_download

from ..data import load_image
from ..utils import get_storage_dir

__all__ = [
    'BackgroundImageSet',
    'list_bg_image_files',
    'get_bg_image_file',
    'get_bg_image',
    'random_bg_image_file',
    'random_bg_image',
]

_BG_REPO = 'deepghs/anime-bg'


@lru_cache()
def _global_df() -> pd.DataFrame:
    return pd.read_csv(hf_hub_download(
        repo_id=_BG_REPO,
        repo_type='dataset',
        filename='images.csv'
    ))


@lru_cache()
def _bg_root_dir() -> str:
    root = os.path.join(get_storage_dir(), 'bg')
    os.makedirs(root, exist_ok=True)
    return root


class BackgroundImageSet:
    def __init__(self, width: Optional[int] = None, height: Optional[int] = None,
                 scale_only: bool = False, strict_level: float = 1.0, min_range: int = 5,
                 min_width: Optional[int] = None, min_height: Optional[int] = None):
        df = _global_df().copy()
        if min_width:
            df = df[df['width'] >= min_width]
        if min_height:
            df = df[df['height'] >= min_height]

        if width and height:
            r1, r2 = width / height, height / width
            df['d1'] = (df['width'] / df['height'] - r1).abs()
            df['d2'] = (df['height'] / df['width'] - r2).abs()
            df['dx'] = df['d1'] + df['d2']
            if scale_only:
                df = df.sort_values(['dx'], ascending=[True])
            else:
                df['dy'] = ((df['width'] * df['height']) ** 0.5 - (width * height) ** 0.5).abs()
                df = df.sort_values(['dx', 'dy'], ascending=[True, True])
            df['i'] = range(len(df))
            idx = df['dx'] < (df['dx'].mean() - df['dx'].std() * strict_level)
            idx = idx | (df['i'] < max(min_range, 1))
            df = df[idx]
        elif width and not height:
            df['dist'] = (df['width'] - width).abs()
            df = df.sort_values(['dist'], ascending=[True])
            df['i'] = range(len(df))
            idx = df['dist'] < (df['dist'].mean() - df['dist'].std() * strict_level)
            idx = idx | (df['i'] < max(min_range, 1))
            df = df[idx]
        elif not width and height:
            df['dist'] = (df['height'] - height).abs()
            df = df.sort_values(['dist'], ascending=[True])
            df['i'] = range(len(df))
            idx = df['dist'] < (df['dist'].mean() - df['dist'].std() * strict_level)
            idx = idx | (df['i'] < max(min_range, 1))
            df = df[idx]
        else:
            pass

        self.df = df
        self._map = {item['filename']: item for item in self.df.to_dict('records')}

    def list_image_files(self) -> List[str]:
        return self.df['filename'].tolist()

    def get_image_file(self, filename: str) -> str:
        return self._load_local_image_file(filename)

    def get_image(self, filename: str) -> Image.Image:
        return load_image(self.get_image_file(filename))

    def random_image_file(self) -> str:
        return self.get_image_file(self.df['filename'].sample(n=1).tolist()[0])

    def random_image(self) -> Image.Image:
        return load_image(self.random_image_file())

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


@lru_cache()
def _get_default_set():
    return BackgroundImageSet()


def list_bg_image_files() -> List[str]:
    return _get_default_set().list_image_files()


def get_bg_image_file(filename: str) -> str:
    return _get_default_set().get_image_file(filename)


def get_bg_image(filename) -> Image.Image:
    return _get_default_set().get_image(filename)


def random_bg_image_file() -> str:
    return _get_default_set().random_image_file()


def random_bg_image() -> Image.Image:
    return _get_default_set().random_image()
