"""
This module provides utilities for downloading and handling images from URLs, with special support for GitHub and Hugging Face URLs.

The module includes functions for:

- Downloading images from URLs with progress tracking
- URL validation and processing
- Special handling for GitHub and Hugging Face hosted images

Main components:

- download_image_from_url: Downloads and returns an image from a given URL
- is_http_url: Checks if a given URL is a valid HTTP/HTTPS URL
- Internal utilities for processing GitHub and Hugging Face URLs
"""

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
    """
    Download an image from a URL and return it as a PIL Image object.

    :param url: URL of the image to download
    :type url: str
    :param silent: If True, suppress progress bar display
    :type silent: bool
    :param expected_size: Expected file size in bytes, used for progress bar
    :type expected_size: Optional[int]
    :param kwargs: Additional keyword arguments passed to the session.get() method

    :return: Downloaded image as PIL Image object
    :rtype: Image.Image

    :raises ValueError: If the URL is not supported (especially for HF URLs)
    :raises requests.RequestException: If download fails
    :raises PIL.UnidentifiedImageError: If downloaded content is not a valid image

    :example:
        >>> image = download_image_from_url('https://example.com/image.jpg')
        >>> image.show()
    """
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
    """
    Check if a given URL is a valid HTTP or HTTPS URL.

    :param url: URL to check
    :type url: str

    :return: True if URL is a valid HTTP/HTTPS URL, False otherwise
    :rtype: bool

    :example:
        >>> is_http_url('https://example.com')
        True
        >>> is_http_url('ftp://example.com')
        False
    """
    if not isinstance(url, str):
        return False

    split = urlsplit(url)
    return split.scheme == 'http' or split.scheme == 'https'


_GITHUB_SUFFIX = {('github', 'com')}


def _is_github_url(url: str) -> bool:
    """
    Check if a URL is a GitHub URL.

    :param url: URL to check
    :type url: str

    :return: True if URL is a GitHub URL, False otherwise
    :rtype: bool
    """
    return tuple(urlsplit(url).host.split('.')[-2:]) in _GITHUB_SUFFIX


def _process_github_url_for_downloading(url: str) -> str:
    """
    Process a GitHub URL to make it suitable for raw file downloading.

    :param url: GitHub URL to process
    :type url: str

    :return: Processed URL for downloading
    :rtype: str
    """
    return str(URLObject(url).add_query_param('raw', 'True'))


_HF_SUFFIX = {('hf', 'co'), ('huggingface', 'co')}


def _is_hf_url(url: str) -> bool:
    """
    Check if a URL is a Hugging Face URL.

    :param url: URL to check
    :type url: str

    :return: True if URL is a Hugging Face URL, False otherwise
    :rtype: bool
    """
    return tuple(urlsplit(url).host.split('.')[-2:]) in _HF_SUFFIX


def _process_hf_url_for_downloading(url: str) -> str:
    """
    Process a Hugging Face URL to make it suitable for file downloading.

    :param url: Hugging Face URL to process
    :type url: str

    :return: Processed URL for downloading
    :rtype: str

    :raises ValueError: If the URL format is not supported
    """
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
