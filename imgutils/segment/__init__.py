"""
Overview:
    Tools for segmenting specific content from images, \
    such as the segmentation of anime characters in the :mod:`imgutils.segment.isnetis` module shown below:

    .. image:: isnetis_trans.plot.py.svg
        :align: center

    This is an overall benchmark of all the segment models:

    .. image:: segment_benchmark.plot.py.svg
        :align: center

"""
from .isnetis import get_isnetis_mask, segment_with_isnetis, segment_rgba_with_isnetis
