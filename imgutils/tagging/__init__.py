"""
Overview:
    Get tags for anime images.

    This is an overall benchmark of all the danbooru models:

    .. image:: tagging_benchmark.plot.py.svg
        :align: center

"""
from .blacklist import is_blacklisted, drop_blacklisted_tags
from .character import is_basic_character_tag, drop_basic_character_tags
from .deepdanbooru import get_deepdanbooru_tags
from .deepgelbooru import get_deepgelbooru_tags
from .format import tags_to_text, add_underline, remove_underline
from .match import tag_match_suffix, tag_match_prefix, tag_match_full
from .mldanbooru import get_mldanbooru_tags
from .order import sort_tags
from .overlap import drop_overlap_tags
from .wd14 import get_wd14_tags, convert_wd14_emb_to_prediction, denormalize_wd14_emb
