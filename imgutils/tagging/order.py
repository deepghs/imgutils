import random
import re
from typing import Literal, Union, List, Mapping


def sort_tags(tags: Union[List[str], Mapping[str, float]],
              mode: Literal['original', 'shuffle', 'score'] = 'score') -> List[str]:
    """
    Sort the input list or mapping of tags by specified mode.

    Tags can represent people counts (e.g., '1girl', '2boys'), and 'solo' tags.

    :param tags: List or mapping of tags to be sorted.
    :type tags: Union[List[str], Mapping[str, float]]
    :param mode: The mode for sorting the tags. Options: 'original' (original order),
                 'shuffle' (random shuffle), 'score' (sorted by score if available).
    :type mode: Literal['original', 'shuffle', 'score']
    :return: Sorted list of tags based on the specified mode.
    :rtype: List[str]
    :raises ValueError: If an unknown sort mode is provided.
    :raises TypeError: If the input tags are of unsupported type or if mode is 'score'
                       and the input is a list (as it does not have scores).

    Examples:
        Sorting tags in original order:

        >>> from imgutils.tagging import sort_tags
        >>>
        >>> tags = ['1girls', 'solo', 'red_hair', 'cat ears']
        >>> sort_tags(tags, mode='original')
        ['solo', '1girls', 'red_hair', 'cat ears']
        >>>
        >>> tags = {'1girls': 0.9, 'solo': 0.95, 'red_hair': 1.0, 'cat_ears': 0.92}
        >>> sort_tags(tags, mode='original')
        ['solo', '1girls', 'red_hair', 'cat_ears']

        Sorting tags by score (for a mapping of tags with scores):

        >>> from imgutils.tagging import sort_tags
        >>>
        >>> tags = {'1girls': 0.9, 'solo': 0.95, 'red_hair': 1.0, 'cat_ears': 0.92}
        >>> sort_tags(tags)
        ['solo', '1girls', 'red_hair', 'cat_ears']

        Shuffling tags (output is not unique)

        >>> from imgutils.tagging import sort_tags
        >>>
        >>> tags = ['1girls', 'solo', 'red_hair', 'cat ears']
        >>> sort_tags(tags, mode='shuffle')
        ['solo', '1girls', 'red_hair', 'cat ears']
        >>>
        >>> tags = {'1girls': 0.9, 'solo': 0.95, 'red_hair': 1.0, 'cat_ears': 0.92}
        >>> sort_tags(tags, mode='shuffle')
        ['solo', '1girls', 'cat_ears', 'red_hair']
    """
    if mode not in {'original', 'shuffle', 'score'}:
        raise ValueError(f'Unknown sort_mode, \'original\', '
                         f'\'shuffle\' or \'score\' expected but {mode!r} found.')
    npeople_tags = []
    remaining_tags = []

    if 'solo' in tags:
        npeople_tags.append('solo')

    for tag in tags:
        if tag == 'solo':
            continue
        if re.fullmatch(r'^\d+\+?(boy|girl)s?$', tag):  # 1girl, 1boy, 2girls, 3boys, 9+girls
            npeople_tags.append(tag)
        else:
            remaining_tags.append(tag)

    if mode == 'score':
        if isinstance(tags, dict):
            remaining_tags = sorted(remaining_tags, key=lambda x: -tags[x])
        else:
            raise TypeError(f'Sort mode {mode!r} not supported for list, '
                            f'for it do not have scores.')
    elif mode == 'shuffle':
        random.shuffle(remaining_tags)
    else:
        pass

    return npeople_tags + remaining_tags
