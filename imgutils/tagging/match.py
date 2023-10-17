import re
from functools import lru_cache
from typing import List, Set, Tuple

from hbutils.string import singular_form, plural_form


@lru_cache(maxsize=2048)
def _cached_singular_form(word: str) -> str:
    """
    Get the singular form of a word.

    :param word: The input word.
    :type word: str
    :return: The singular form of the word.
    :rtype: str
    """
    return singular_form(word)


@lru_cache(maxsize=2048)
def _cache_plural_form(word: str) -> str:
    """
    Get the plural form of a word.

    :param word: The input word.
    :type word: str
    :return: The plural form of the word.
    :rtype: str
    """
    return plural_form(word)


def _split_to_words(text: str) -> List[str]:
    """
    Split a string into words and return them in lowercase.

    :param text: The input text to split.
    :type text: str
    :return: List of lowercase words.
    :rtype: List[str]
    """
    return [word.lower() for word in re.split(r'[\s_]+', text) if word]


def _words_to_matcher(words: List[str], enable_forms: bool = True) -> Set[Tuple[str, ...]]:
    """
    Generate word matchers based on single, plural, and singular forms.

    :param words: List of words.
    :type words: List[str]
    :param enable_forms: Enable singular and plural forms. Default is True.
    :type enable_forms: bool
    :return: Set of word matchers.
    :rtype: Set[Tuple[str, ...]]
    """
    if words and enable_forms:
        words_tuples = [
            words,
            [*words[:-1], _cached_singular_form(words[-1])],
            [*words[:-1], _cache_plural_form(words[-1])],
        ]
    else:
        words_tuples = [words]
    return set([tuple(item) for item in words_tuples])


def tag_match_suffix(text: str, suffix: str) -> bool:
    """
    Check if a text matches a given suffix.

    :param text: The input text.
    :type text: str
    :param suffix: The suffix to match.
    :type suffix: str
    :return: True if the text matches the suffix, False otherwise.
    :rtype: bool
    """
    _suffix_words = _split_to_words(suffix)
    _text_words = _split_to_words(text)
    if not _suffix_words:
        return True

    _text_words = _text_words[-len(_suffix_words):]
    return bool(_words_to_matcher(_text_words) & _words_to_matcher(_suffix_words))


def tag_match_prefix(text: str, prefix: str) -> bool:
    """
    Check if a text matches a given prefix.

    :param text: The input text.
    :type text: str
    :param prefix: The prefix to match.
    :type prefix: str
    :return: True if the text matches the prefix, False otherwise.
    :rtype: bool
    """
    _prefix_words = _split_to_words(prefix)
    _text_words = _split_to_words(text)
    if not _prefix_words:
        return True

    _text_words = _text_words[:len(_prefix_words)]
    return bool(
        _words_to_matcher(_text_words, enable_forms=False) &
        _words_to_matcher(_prefix_words, enable_forms=False)
    )


def tag_match_full(t1: str, t2: str) -> bool:
    """
    Check if two texts match each other fully.

    :param t1: The first text.
    :type t1: str
    :param t2: The second text.
    :type t2: str
    :return: True if the texts match fully, False otherwise.
    :rtype: bool
    """
    _t1_words = _split_to_words(t1)
    _t2_words = _split_to_words(t2)
    return bool(_words_to_matcher(_t1_words) & _words_to_matcher(_t2_words))
