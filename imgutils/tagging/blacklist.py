from functools import lru_cache
from typing import Union, List, Mapping, Set, Optional

from huggingface_hub import hf_hub_download


@lru_cache()
def _load_online_blacklist() -> List[str]:
    with open(hf_hub_download(
            'alea31415/tag_filtering',
            'blacklist_tags.txt',
            repo_type='dataset',
    ), 'r') as f:
        return [line.strip() for line in f if line.strip()]


def _is_blacklisted(tag: str, blacklist: Set[str]):
    return (tag in blacklist or
            tag.replace('_', ' ') in blacklist or
            tag.replace(' ', '_') in blacklist)


@lru_cache()
def _online_blacklist_set() -> Set[str]:
    return set(_load_online_blacklist())


def is_blacklisted(tags: str):
    return _is_blacklisted(tags, _online_blacklist_set())


def drop_blacklisted_tags(tags: Union[List[str], Mapping[str, float]],
                          use_presets: bool = True, custom_blacklist: Optional[List[str]] = None) \
        -> Union[List[str], Mapping[str, float]]:
    blacklist = []
    if use_presets:
        blacklist.extend(_load_online_blacklist())
    blacklist.extend(custom_blacklist or [])

    blacklist = set(tag.replace(' ', '_') for tag in blacklist)
    blacklist_update = set(tag.replace('_', ' ') for tag in blacklist)
    blacklist.update(blacklist_update)

    if isinstance(tags, dict):
        return {tag: value for tag, value in tags.items() if not _is_blacklisted(tag, blacklist)}
    elif isinstance(tags, list):
        return [tag for tag in tags if not _is_blacklisted(tag, blacklist)]
    else:
        raise TypeError(f"Unsupported types of tags, dict or list expected, but {tags!r} found.")
