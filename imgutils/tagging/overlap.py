import json
from functools import lru_cache
from typing import Mapping, List

from huggingface_hub import hf_hub_download


@lru_cache()
def _get_overlap_tags() -> Mapping[str, List[str]]:
    """
    Retrieve the overlap tag information from the specified Hugging Face Hub repository.

    This function downloads a JSON file containing tag overlap information and parses it into a dictionary.

    :return: A dictionary where keys are tags and values are lists of overlapping tags.
    :rtype: Mapping[str, List[str]]
    """
    json_file = hf_hub_download(
        'alea31415/tag_filtering',
        'overlap_tags.json',
        repo_type='dataset',
    )
    with open(json_file, 'r') as file:
        data = json.load(file)

    return {
        entry['query']: entry['has_overlap']
        for entry in data if 'has_overlap' in entry and entry['has_overlap']
    }


def drop_overlap_tags(tags: List[str]) -> List[str]:
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
    """
    overlap_tags_dict = _get_overlap_tags()
    result_tags = []
    tags_underscore = [tag.replace(' ', '_') for tag in tags]

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

    return result_tags


def drop_overlaps_for_dict(tags: Mapping[str, float]) -> Mapping[str, float]:
    """
    Drop overlapping tags from the given dictionary of tags with confidence scores.

    This function removes tags that have overlaps with other tags based on precomputed overlap information.

    :param tags: A dictionary where keys are tags and values are confidence scores.
    :type tags: Mapping[str, float]
    :return: A dictionary with non-overlapping tags and their corresponding confidence scores.
    :rtype: Mapping[str, float]

    Examples::
        >>> from imgutils.tagging import drop_overlaps_for_dict
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
        >>> drop_overlaps_for_dict(tags)
        {
            '1girl': 0.8849405313291128,
            'solo': 0.8548297594823425,
            'very_long_hair': 0.6615180440330748,
            'red_hair': 0.21552028866308015,
            'medium_breasts': 0.47744464927382957
        }
    """
    key_set = set(drop_overlap_tags(list(tags.keys())))
    return {tag: confidence for tag, confidence in tags.items() if tag in key_set}
