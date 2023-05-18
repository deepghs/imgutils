"""
Overview:
    Obtaining the outline (or you can call that line drawing) of an anime image.

    Here is the example and comparison:

    .. image:: edge_demo.plot.py.svg
        :align: center

    This is an overall benchmark of all the outline models:

    .. image:: edge_benchmark.plot.py.svg
        :align: center

"""
from .canny import get_edge_by_canny, edge_image_with_canny
from .lineart import get_edge_by_lineart, edge_image_with_lineart
from .lineart_anime import get_edge_by_lineart_anime, edge_image_with_lineart_anime
