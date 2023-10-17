"""
Overview:
    Detect and drop character-related basic tags.
"""
from typing import Union, List, Mapping, Tuple, Dict, Set, Optional

from .match import _split_to_words, _words_to_matcher

CHAR_WHITELIST = [
    'drill', 'pubic_hair', 'closed_eyes', 'half-closed_eyes', 'empty_eyes'
]
CHAR_SUFFIXES = [
    'eyes', 'skin', 'hair', 'bun', 'bangs', 'cut', 'sidelocks',
    'twintails', 'braid', 'braids', 'afro', 'ahoge', 'drill',
    'drills', 'bald', 'dreadlocks', 'side up', 'ponytail', 'updo',
    'beard', 'mustache', 'pointy ears', 'ear', 'horn',
]
CHAR_PREFIXES = [
    'hair over', 'hair between'
]

_WordTupleTyping = Tuple[str, ...]


class _SuffixPool:
    """
    Helper class to manage suffixes for character tags.
    """

    def __init__(self, suffixes: Optional[List[str]] = None):
        """
        Initialize a SuffixPool instance.

        :param suffixes: A list of suffixes to include, defaults to None
        :type suffixes: Optional[List[str]], optional
        """
        self._suffixes: Dict[int, Set[_WordTupleTyping]] = {}
        for suffix in (suffixes or []):
            self._append(suffix)

    def _append(self, text: str):
        """
        Append a suffix to the pool.

        :param text: The suffix to append
        :type text: str
        """
        for item in _words_to_matcher(_split_to_words(text)):
            if len(item) not in self._suffixes:
                self._suffixes[len(item)] = set()
            self._suffixes[len(item)].add(item)

    def __contains__(self, text: str):
        """
        Check if a given text contains any suffix from the pool.

        :param text: The text to check
        :type text: str
        :return: True if the text contains a suffix, False otherwise
        :rtype: bool
        """
        words = _split_to_words(text)
        for length, tpl_set in self._suffixes.items():
            if length > len(words):
                continue

            seg = [] if length == 0 else words[-length:]
            if _words_to_matcher(seg) & tpl_set:
                return True

        return False


class _PrefixPool:
    """
    Helper class to manage prefixes for character tags.
    """

    def __init__(self, prefixes: Optional[List[str]] = None):
        """
        Initialize a PrefixPool instance.

        :param prefixes: A list of prefixes to include, defaults to None
        :type prefixes: Optional[List[str]], optional
        """
        self._prefixes: Dict[int, Set[_WordTupleTyping]] = {}
        for prefix in (prefixes or []):
            self._append(prefix)

    def _append(self, text: str):
        """
        Append a prefix to the pool.

        :param text: The prefix to append
        :type text: str
        """
        for item in _words_to_matcher(_split_to_words(text), enable_forms=False):
            if len(item) not in self._prefixes:
                self._prefixes[len(item)] = set()
            self._prefixes[len(item)].add(item)

    def __contains__(self, text: str):
        """
        Check if a given text contains any prefix from the pool.

        :param text: The text to check
        :type text: str
        :return: True if the text contains a prefix, False otherwise
        :rtype: bool
        """
        words = _split_to_words(text)
        for length, tpl_set in self._prefixes.items():
            if length > len(words):
                continue

            seg = words[:length]
            if _words_to_matcher(seg, enable_forms=False) & tpl_set:
                return True

        return False


class CharacterTagPool:
    """
    A pool of character-related tags for detection and removal of basic character tags.
    """

    def __init__(self, whitelist: Optional[List[str]] = None,
                 suffixes: Optional[List[str]] = None,
                 prefixes: Optional[List[str]] = None):
        """
        Initialize a CharacterTagPool instance.

        :param whitelist: A list of whitelisted tags, defaults to None
        :type whitelist: Optional[List[str]], optional
        :param suffixes: A list of suffixes to consider, defaults to None
        :type suffixes: Optional[List[str]], optional
        :param prefixes: A list of prefixes to consider, defaults to None
        :type prefixes: Optional[List[str]], optional
        """
        self._whitelist = _SuffixPool(whitelist or CHAR_WHITELIST)
        self._suffixes = _SuffixPool(suffixes or CHAR_SUFFIXES)
        self._prefixes = _PrefixPool(prefixes or CHAR_PREFIXES)

    def is_basic_character_tag(self, tag: str) -> bool:
        """
        Check if a given tag is a basic character tag.

        :param tag: The tag to check
        :type tag: str
        :return: True if the tag is a basic character tag, False otherwise
        :rtype: bool
        """
        if tag in self._whitelist:
            return False
        else:
            return (tag in self._suffixes) or (tag in self._prefixes)

    def drop_basic_character_tags(self, tags: Union[List[str], Mapping[str, float]]) \
            -> Union[List[str], Mapping[str, float]]:
        """
        Drop basic character tags from a list or mapping of tags.

        :param tags: The tags to process
        :type tags: Union[List[str], Mapping[str, float]]
        :return: Processed tags with basic character tags removed
        :rtype: Union[List[str], Mapping[str, float]]
        """
        if isinstance(tags, dict):
            return {tag: value for tag, value in tags.items() if not self.is_basic_character_tag(tag)}
        elif isinstance(tags, list):
            return [tag for tag in tags if not self.is_basic_character_tag(tag)]
        else:
            raise TypeError(f"Unsupported types of tags, dict or list expected, but {tags!r} found.")


_DEFAULT_CHARACTER_POOL = CharacterTagPool()


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
    return _DEFAULT_CHARACTER_POOL.is_basic_character_tag(tag)


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
    return _DEFAULT_CHARACTER_POOL.drop_basic_character_tags(tags)
