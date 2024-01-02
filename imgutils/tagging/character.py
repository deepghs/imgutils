"""
Overview:
    Detect and drop character-related basic tags.
"""
from typing import Union, List, Mapping, Tuple, Dict, Set, Optional

from .match import _split_to_words, _words_to_matcher

CHAR_WHITELIST_SUFFIX = [
    'anal_hair',
    'anal_tail',
    'arm_behind_head',
    'arm_hair',
    'arm_under_breasts',
    'arms_behind_head',
    'bird_on_head',
    'blood_in_hair',
    'breasts_on_glass',
    'breasts_on_head',
    'cat_on_head',
    'closed_eyes',
    'clothed_female_nude_female',
    'clothed_female_nude_male',
    'clothed_male_nude_female',
    'clothes_between_breasts',
    'cream_on_face',
    'drying_hair',
    'empty_eyes',
    'face_to_breasts',
    'facial',
    'food_on_face',
    'food_on_head',
    'game_boy',
    "grabbing_another's_hair",
    'grabbing_own_breast',
    'gun_to_head',
    'half-closed_eyes',
    'head_between_breasts',
    'heart_in_eye',
    'multiple_boys',
    'multiple_girls',
    'object_on_breast',
    'object_on_head',
    'paint_splatter_on_face',
    'parted_lips',
    'penis_on_face',
    'person_on_head',
    'pokemon_on_head',
    'pubic_hair',
    'rabbit_on_head',
    'rice_on_face',
    'severed_head',
    'star_in_eye',
    'sticker_on_face',
    'tentacles_on_male',
    'tying_hair'
]
CHAR_WHITELIST_PREFIX = [
    'holding', 'hand on', 'hands on', 'hand to', 'hands to',
    'hand in', 'hands in', 'hand over', 'hands over',
    'futa with', 'futa on', 'cum on', 'covering', 'adjusting', 'rubbing',
    'sitting', 'shading', 'playing', 'cutting',
]
CHAR_WHITELIST_WORD = [
    'drill',
]
CHAR_SUFFIXES = [
    'eyes', 'skin', 'hair', 'bun', 'bangs', 'cut', 'sidelocks',
    'twintails', 'braid', 'braids', 'afro', 'ahoge', 'drill',
    'drills', 'bald', 'dreadlocks', 'side up', 'ponytail', 'updo',
    'beard', 'mustache', 'pointy ears', 'ear', 'horn', 'tail', 'wing',
    'ornament', 'hairband', 'pupil', 'bow', 'eyewear', 'headwear',
    'ribbon', 'crown', 'cap', 'hat', 'hairclip', 'breast', 'mole',
    'halo', 'earrings', 'animal ear fluff', 'hair flower', 'glasses',
    'fang', 'female', 'girl', 'boy', 'male', 'beret', 'heterochromia',
    'headdress', 'headgear', 'eyepatch', 'headphones', 'eyebrows', 'eyelashes',
    'sunglasses', 'hair intakes', 'scrunchie', 'ear_piercing', 'head',
    'on face', 'on head', 'on hair', 'headband', 'hair rings', 'under_mouth',
    'freckles', 'lip', 'eyeliner', 'eyeshadow', 'tassel', 'over one eye',
    'drill', 'drill hair',
]
CHAR_PREFIXES = [
    'hair over', 'hair between', 'facial',
]

_WordTupleTyping = Tuple[str, ...]


class _WordPool:
    """
    Helper class to manage  character tags.
    """

    def __init__(self, words: Optional[List[str]] = None):
        """
        Initialize a _WordPool instance.

        :param words: A list of words to include, defaults to None
        :type words: Optional[List[str]], optional
        """
        self._words: Dict[int, Set[_WordTupleTyping]] = {}
        for word in (words or []):
            self._append(word)

    def _append(self, text: str):
        """
        Append a word to the pool.

        :param text: The word to append
        :type text: str
        """
        for item in _words_to_matcher(_split_to_words(text)):
            if len(item) not in self._words:
                self._words[len(item)] = set()
            self._words[len(item)].add(item)

    def __contains__(self, text: str):
        """
        Check if a given text equals to any word from the pool.

        :param text: The text to check
        :type text: str
        :return: True if the text equals to a word, False otherwise
        :rtype: bool
        """
        words = tuple(_split_to_words(text))
        return len(words) in self._words and words in self._words[len(words)]


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

    def __init__(
            self,
            whitelist_suffixes: Optional[List[str]] = None,
            whitelist_prefixes: Optional[List[str]] = None,
            whitelist_words: Optional[List[str]] = None,
            suffixes: Optional[List[str]] = None,
            prefixes: Optional[List[str]] = None
    ):
        """
        Initialize a CharacterTagPool instance.

        :param whitelist_suffixes: A list of whitelisted suffixes, defaults to None
        :type whitelist_suffixes: Optional[List[str]], optional
        :param suffixes: A list of suffixes to consider, defaults to None
        :type suffixes: Optional[List[str]], optional
        :param prefixes: A list of prefixes to consider, defaults to None
        :type prefixes: Optional[List[str]], optional
        """
        self._whitelist_suffix = _SuffixPool(whitelist_suffixes or CHAR_WHITELIST_SUFFIX)
        self._whitelist_prefix = _PrefixPool(whitelist_prefixes or CHAR_WHITELIST_PREFIX)
        self._whitelist_words = _WordPool(whitelist_words or CHAR_WHITELIST_WORD)
        self._suffixes = _SuffixPool(suffixes or CHAR_SUFFIXES)
        self._prefixes = _PrefixPool(prefixes or CHAR_PREFIXES)

    def _is_in_whitelist(self, tag: str) -> bool:
        return (tag in self._whitelist_words) or (tag in self._whitelist_suffix) or (tag in self._whitelist_prefix)

    def _is_in_common(self, tag: str) -> bool:
        return (tag in self._suffixes) or (tag in self._prefixes)

    def is_basic_character_tag(self, tag: str) -> bool:
        """
        Check if a given tag is a basic character tag.

        :param tag: The tag to check
        :type tag: str
        :return: True if the tag is a basic character tag, False otherwise
        :rtype: bool
        """
        if self._is_in_whitelist(tag):
            return False
        else:
            return self._is_in_common(tag)

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
