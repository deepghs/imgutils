"""
Overview:
    Detect and drop some blacklisted tags, which are listed `here <https://huggingface.co/datasets/alea31415/tag_filtering/blob/main/blacklist_tags.txt>`_.
"""
from typing import Union, List, Mapping, Set, Optional, Tuple

from huggingface_hub import hf_hub_download

from .match import _words_to_matcher, _split_to_words
from ..utils import ts_lru_cache


@ts_lru_cache()
def _load_online_blacklist() -> List[str]:
    """
    Load the online blacklist tags from the specified dataset repository.

    :return: List of blacklisted tags.
    :rtype: List[str]
    """
    with open(hf_hub_download(
            'alea31415/tag_filtering',
            'blacklist_tags.txt',
            repo_type='dataset',
    ), 'r') as f:
        return [line.strip() for line in f if line.strip()]


@ts_lru_cache()
def _online_blacklist_set() -> Set[Tuple[str, ...]]:
    """
    Get the online blacklist as a set.

    :return: Set of blacklisted tags.
    :rtype: Set[str]
    """
    set_ = set()
    for tag in _load_online_blacklist():
        set_ = set_ | _words_to_matcher(_split_to_words(tag))
    return set_


def _is_blacklisted(tag: str, blacklist_set: Set[Tuple[str, ...]]) -> bool:
    _tag_matcher = _words_to_matcher(_split_to_words(tag))
    return bool(set(_tag_matcher & blacklist_set))


def is_blacklisted(tag: str) -> bool:
    """
    Check if any of the given tags are blacklisted.

    :param tag: Tags to be checked.
    :type tag: str
    :return: True if any tag is blacklisted, False otherwise.
    :rtype: bool

    Examples::
        >>> from imgutils.tagging import is_blacklisted
        >>>
        >>> is_blacklisted('cosplay')
        True
        >>> is_blacklisted('no_eyewear')
        True
        >>> is_blacklisted('no eyewear')  # span does not matter
        True
        >>> is_blacklisted('red_hair')
        False
    """
    return _is_blacklisted(tag, _online_blacklist_set())


def drop_blacklisted_tags(tags: Union[List[str], Mapping[str, float]],
                          use_presets: bool = True, custom_blacklist: Optional[List[str]] = None) \
        -> Union[List[str], Mapping[str, float]]:
    """
    Drop blacklisted tags from the given list or mapping of tags.

    :param tags: List or mapping of tags to be filtered.
    :type tags: Union[List[str], Mapping[str, float]]
    :param use_presets: Whether to use the online blacklist presets, defaults to True.
    :type use_presets: bool, optional
    :param custom_blacklist: Custom blacklist to be used, defaults to None.
    :type custom_blacklist: Optional[List[str]], optional
    :return: Filtered list or mapping of tags without the blacklisted ones.
    :rtype: Union[List[str], Mapping[str, float]]
    :raises TypeError: If the input tags are neither a list nor a dictionary.

    Examples::
        >>> from imgutils.tagging import drop_blacklisted_tags
        >>>
        >>> drop_blacklisted_tags({
        ...     'solo': 1.0, '1girl': 0.95,
        ...     'cosplay': 0.7, 'no_eyewear': 0.6,
        ... })
        {'solo': 1.0, '1girl': 0.95}
        >>> drop_blacklisted_tags(['solo', '1girl', 'cosplay', 'no_eyewear'])
        ['solo', '1girl']
    """
    blacklist = set()
    if use_presets:
        blacklist = blacklist | _online_blacklist_set()
    for tag in (custom_blacklist or []):
        blacklist = blacklist | _words_to_matcher(_split_to_words(tag))

    if isinstance(tags, dict):
        return {tag: value for tag, value in tags.items() if not _is_blacklisted(tag, blacklist)}
    elif isinstance(tags, list):
        return [tag for tag in tags if not _is_blacklisted(tag, blacklist)]
    else:
        raise TypeError(f"Unsupported types of tags, dict or list expected, but {tags!r} found.")
