"""
Overview:
    A tool that maps tags to a single string, commonly used for image feature labeling during model training.
"""
import re
from typing import Mapping

RE_SPECIAL = re.compile(r'([\\()])')


def tags_to_text(tags: Mapping[str, float],
                 use_spaces: bool = False, use_escape: bool = True,
                 include_score: bool = False, score_descend: bool = True) -> str:
    """
    Overview:
        Transform tags to text for data labeling.
    
    :param tags: Tags.
    :param use_spaces: Use whitespace instead of ``_`` in text, default is ``False``.
    :param use_escape: Escape unsafe characters in text, such as ``(`` to ``\\(``, default is ``True``.
    :param include_score: Add score in text, default is ``False``.
    :param score_descend: Sort in descending order by the score of each tag. Default is ``True``.

    Examples::
        >>> from imgutils.tagging import tags_to_text
        >>>
        >>> # a group of tags
        >>> tags = {
        ...     'panty_pull': 0.6826801300048828,
        ...     'panties': 0.958938717842102,
        ...     'drinking_glass': 0.9340789318084717,
        ...     'areola_slip': 0.41196826100349426,
        ...     '1girl': 0.9988248348236084,
        ... }
        >>>
        >>> tags_to_text(tags)
        '1girl, panties, drinking_glass, panty_pull, areola_slip'
        >>> tags_to_text(tags, use_spaces=True)
        '1girl, panties, drinking glass, panty pull, areola slip'
        >>> tags_to_text(tags, include_score=True)
        '(1girl:0.999), (panties:0.959), (drinking_glass:0.934), (panty_pull:0.683), (areola_slip:0.412)'
    """
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
        if include_score:
            t_text = f"({t_text}:{score:.3f})"
        text_items.append(t_text)

    return ', '.join(text_items)
