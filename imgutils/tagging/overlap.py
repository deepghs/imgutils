import copy
import json
from typing import Mapping, List, Union

from huggingface_hub import hf_hub_download

from ..utils import ts_lru_cache


@ts_lru_cache()
def _get_overlap_tags() -> Mapping[str, List[str]]:
    """
    Retrieve the overlap tag information from the specified Hugging Face Hub repository.

    This function downloads a JSON file containing tag overlap information and parses it into a dictionary.

    :return: A dictionary where keys are tags and values are lists of overlapping tags.
    :rtype: Mapping[str, List[str]]
    """
    json_file = hf_hub_download(
        'alea31415/tag_filtering',
        'overlap_tags_simplified.json',
        repo_type='dataset',
    )
    with open(json_file, 'r') as file:
        data = json.load(file)

    return data


def drop_overlap_tags(tags: Union[List[str], Mapping[str, float]]) -> Union[List[str], Mapping[str, float]]:
    """
    Drop overlapping tags from the given list of tags.

    This function removes tags that have overlaps with other tags based on precomputed overlap information.

    :param tags: A list of tags.
    :type tags: List[str]
    :return: A list of tags without overlaps.
    :rtype: List[str]

    Examples::
        >>> from imgutils.tagging import drop_overlap_tags
        >>>
        >>> tags = [
        ...     '1girl', 'solo',
        ...     'long_hair', 'very_long_hair', 'red_hair',
        ...     'breasts', 'medium_breasts',
        ... ]
        >>> drop_overlap_tags(tags)
        ['1girl', 'solo', 'very_long_hair', 'red_hair', 'medium_breasts']
        >>>
        >>> tags = {
        ...     '1girl': 0.8849405313291128,
        ...     'solo': 0.8548297594823425,
        ...     'long_hair': 0.03910296474461261,
        ...     'very_long_hair': 0.6615180440330748,
        ...     'red_hair': 0.21552028866308015,
        ...     'breasts': 0.3165260620737027,
        ...     'medium_breasts': 0.47744464927382957,
        ... }
        >>> drop_overlap_tags(tags)
        {
            '1girl': 0.8849405313291128,
            'solo': 0.8548297594823425,
            'very_long_hair': 0.6615180440330748,
            'red_hair': 0.21552028866308015,
            'medium_breasts': 0.47744464927382957
        }
    """
    overlap_tags_dict = _get_overlap_tags()
    result_tags = []
    _origin_tags = copy.deepcopy(tags)
    if isinstance(tags, dict):
        tags = list(tags.keys())
    tags_underscore = [tag.replace(' ', '_') for tag in tags]

    tags: List[str]
    tags_underscore: List[str]
    for tag, tag_ in zip(tags, tags_underscore):
        to_remove = False

        # Case 1: If the tag is a key and some of the associated values are in tags
        if tag_ in overlap_tags_dict:
            overlap_values = set(val for val in overlap_tags_dict[tag_])
            if overlap_values.intersection(set(tags_underscore)):
                to_remove = True

        # Checking superword condition separately
        for tag_another in tags:
            if tag in tag_another and tag != tag_another:
                to_remove = True
                break

        if not to_remove:
            result_tags.append(tag)

    if isinstance(_origin_tags, list):
        return result_tags
    elif isinstance(_origin_tags, dict):
        _rtags_set = set(result_tags)
        return {key: value for key, value in _origin_tags.items() if key in _rtags_set}
    else:
        raise TypeError(f'Unknown tags type - {_origin_tags!r}.')  # pragma: no cover
