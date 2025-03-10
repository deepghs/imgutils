import io
from typing import Optional

import pyrfc6266
from PIL import Image
from hbutils.system import urlsplit
from huggingface_hub import get_session
from tqdm import tqdm
from urlobject import URLObject

__all__ = [
    'download_image_from_url',
    'is_http_url',
]


def download_image_from_url(url: str, silent: bool = False, expected_size: Optional[int] = None,
                            **kwargs) -> Image.Image:
    if _is_github_url(url):
        url = _process_github_url_for_downloading(url)
    elif _is_hf_url(url):
        url = _process_hf_url_for_downloading(url)

    session = get_session()
    with session.get(url, stream=True, allow_redirects=True, **kwargs) as response:
        expected_size = expected_size or response.headers.get('Content-Length', None)
        expected_size = int(expected_size) if expected_size is not None else expected_size
        filename = None
        if response.headers.get('Content-Disposition'):
            filename = pyrfc6266.parse_filename(response.headers.get('Content-Disposition'))
        filename = filename or urlsplit(url).filename

        with io.BytesIO() as bf:
            with tqdm(total=expected_size, unit='B', unit_scale=True, unit_divisor=1024,
                      desc=filename, disable=silent) as pbar:
                for chunk in response.iter_content(chunk_size=1024):
                    bf.write(chunk)
                    pbar.update(len(chunk))

            bf.seek(0)
            image = Image.open(bf)
            image.load()
            return image


def is_http_url(url: str) -> bool:
    if not isinstance(url, str):
        return False

    split = urlsplit(url)
    return split.scheme == 'http' or split.scheme == 'https'


_GITHUB_SUFFIX = {('github', 'com')}


def _is_github_url(url: str) -> bool:
    # assume that is_http_url(url) is True
    return tuple(urlsplit(url).host.split('.')[-2:]) in _GITHUB_SUFFIX


def _process_github_url_for_downloading(url: str) -> str:
    return str(URLObject(url).with_query('raw=True'))


_HF_SUFFIX = {('hf', 'co'), ('huggingface', 'co')}


def _is_hf_url(url: str) -> bool:
    # assume that is_http_url(url) is True
    return tuple(urlsplit(url).host.split('.')[-2:]) in _HF_SUFFIX


def _process_hf_url_for_downloading(url: str) -> str:
    split = urlsplit(url)
    segments = split.path_segments
    if len(segments) >= 2 and (segments[1] == 'datasets' or segments[1] == 'spaces'):
        position = 4
    else:
        position = 3

    if len(segments) > position and segments[position] == 'blob':
        segments = [*segments[:position], 'resolve', *segments[position + 1:]]
    elif len(segments) > position and segments[position] == 'resolve':
        pass
    else:
        raise ValueError(f'Unsupported huggingface URL - {url!r}.')
    return f'{split.scheme}://{split.host}{"/".join(segments)}'
