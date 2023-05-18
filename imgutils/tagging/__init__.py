"""
Overview:
    Get tags for anime images.

    This is an overall benchmark of all the danbooru models:

    .. image:: tagging_benchmark.plot.py.svg
        :align: center

"""
from .deepdanbooru import get_deepdanbooru_tags
from .format import tags_to_text
from .mldanbooru import get_mldanbooru_tags
from .wd14 import get_wd14_tags
