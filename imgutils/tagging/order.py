import random
import re
from typing import Union, List, Mapping

try:
    from typing import Literal
except (ImportError, ModuleNotFoundError):
    from typing_extensions import Literal


def sort_tags(tags: Union[List[str], Mapping[str, float]],
              mode: Literal['original', 'shuffle', 'score'] = 'score') -> List[str]:
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
