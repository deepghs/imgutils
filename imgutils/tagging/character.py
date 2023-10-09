import re
from typing import Union, List, Mapping

from hbutils.string import singular_form, plural_form

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


def _split_to_words(text: str) -> List[str]:
    return [word.lower() for word in re.split(r'[\W_]+', text) if word]


def _match_suffix(tag: str, suffix: str):
    tag_words = _split_to_words(tag)
    suffix_words = _split_to_words(suffix)
    all_suffixes = [suffix_words]
    all_suffixes.append([*suffix_words[:-1], singular_form(suffix_words[0])])
    all_suffixes.append([*suffix_words[:-1], plural_form(suffix_words[0])])

    for suf in all_suffixes:
        if tag_words[-len(suf):] == suf:
            return True

    return False


def _match_prefix(tag: str, prefix: str):
    tag_words = _split_to_words(tag)
    prefix_words = _split_to_words(prefix)
    return tag_words[:len(prefix_words)] == prefix_words


def _match_same(tag: str, expected: str):
    a = _split_to_words(tag)
    as_ = [a, [*a[:-1], singular_form(a[-1])], [*a[:-1], plural_form(a[-1])]]
    as_ = set([tuple(item) for item in as_])

    b = _split_to_words(expected)
    bs_ = [b, [*b[:-1], singular_form(b[-1])], [*b[:-1], plural_form(b[-1])]]
    bs_ = set([tuple(item) for item in bs_])

    return bool(as_ & bs_)


def is_basic_character_tag(tag: str) -> bool:
    if any(_match_same(tag, wl_tag) for wl_tag in _CHAR_WHITELIST):
        return False
    else:
        return (any(_match_suffix(tag, suffix) for suffix in _CHAR_SUFFIXES)
                or any(_match_prefix(tag, prefix) for prefix in _CHAR_PREFIXES))


def drop_basic_character_tags(tags: Union[List[str], Mapping[str, float]]) -> Union[List[str], Mapping[str, float]]:
    if isinstance(tags, dict):
        return {tag: value for tag, value in tags.items() if not is_basic_character_tag(tag)}
    elif isinstance(tags, list):
        return [tag for tag in tags if not is_basic_character_tag(tag)]
    else:
        raise TypeError(f"Unsupported types of tags, dict or list expected, but {tags!r} found.")
