import re
from typing import Mapping

RE_SPECIAL = re.compile(r'([\\()])')


def tags_to_text(tags: Mapping[str, float],
                 use_spaces: bool = False, use_escape: bool = True,
                 include_ranks: bool = False, score_descend: bool = True) -> str:
    text_items = []
    tags_pairs = tags.items()
    if score_descend:
        tags_pairs = sorted(tags_pairs, key=lambda x: (-x[1], x[0]))
    for tag, score in tags_pairs:
        t_text = tag
        if use_spaces:
            t_text = t_text.replace('_', ' ')
        if use_escape:
            t_text = re.sub(RE_SPECIAL, r'\\\1', t_text)
        if include_ranks:
            t_text = f"({t_text}:{score:.3f})"
        text_items.append(t_text)

    return ', '.join(text_items)
