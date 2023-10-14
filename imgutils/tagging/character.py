"""
Overview:
    Detect and drop character-related basic tags.
"""
from typing import Union, List, Mapping

from .match import tag_match_full, tag_match_prefix, tag_match_suffix

_CHAR_WHITELIST = [
    'drill', 'pubic_hair', 'closed_eyes', 'half-closed_eyes', 'empty_eyes'
]
_CHAR_SUFFIXES = [
    'eyes', 'skin', 'hair', 'bun', 'bangs', 'cut', 'sidelocks',
    'twintails', 'braid', 'braids', 'afro', 'ahoge', 'drill',
    'drills', 'bald', 'dreadlocks', 'side up', 'ponytail', 'updo',
    'beard', 'mustache', 'pointy ears', 'ear', 'horn',
]
_CHAR_PREFIXES = [
    'hair over', 'hair between'
]


def is_basic_character_tag(tag: str) -> bool:
    """
    Check if a tag is a basic character tag by matching with predefined whitelisted and blacklisted patterns.

    :param tag: The tag to check.
    :type tag: str
    :return: True if the tag is a basic character tag, False otherwise.
    :rtype: bool

    Examples::
        >>> from imgutils.tagging import is_basic_character_tag
        >>>
        >>> is_basic_character_tag('red hair')
        True
        >>> is_basic_character_tag('red_hair')  # span doesn't matter
        True
        >>> is_basic_character_tag('cat ears')  # singular
        True
        >>> is_basic_character_tag('cat ear')  # plural
        True
        >>> is_basic_character_tag('chair')  # only whole word will be matched
        False
        >>> is_basic_character_tag('hear')  # only whole word will be matched
        False
        >>> is_basic_character_tag('dress')
        False
    """
    if any(tag_match_full(tag, wl_tag) for wl_tag in _CHAR_WHITELIST):
        return False
    else:
        return (any(tag_match_suffix(tag, suffix) for suffix in _CHAR_SUFFIXES)
                or any(tag_match_prefix(tag, prefix) for prefix in _CHAR_PREFIXES))


def drop_basic_character_tags(tags: Union[List[str], Mapping[str, float]]) -> Union[List[str], Mapping[str, float]]:
    """
    Drop basic character tags from the given list or mapping of tags.

    :param tags: List or mapping of tags to be filtered.
    :type tags: Union[List[str], Mapping[str, float]]
    :return: Filtered list or mapping of tags without the basic character tags.
    :rtype: Union[List[str], Mapping[str, float]]
    :raises TypeError: If the input tags are neither a list nor a dictionary.

    Examples::
        >>> from imgutils.tagging import drop_basic_character_tags
        >>>
        >>> drop_basic_character_tags({
        ...     '1girl': 1.0, 'solo': 0.95,
        ...     'red_hair': 0.7, 'cat ears': 0.6,
        ...     'chair': 0.86, 'hear': 0.72,
        ... })
        {'1girl': 1.0, 'solo': 0.95, 'chair': 0.86, 'hear': 0.72}
        >>> drop_basic_character_tags([
        ...     '1girl', 'solo', 'red_hair', 'cat ears', 'chair', 'hear'
        ... ])
        ['1girl', 'solo', 'chair', 'hear']
    """
    if isinstance(tags, dict):
        return {tag: value for tag, value in tags.items() if not is_basic_character_tag(tag)}
    elif isinstance(tags, list):
        return [tag for tag in tags if not is_basic_character_tag(tag)]
    else:
        raise TypeError(f"Unsupported types of tags, dict or list expected, but {tags!r} found.")
