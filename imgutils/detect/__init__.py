"""
Overview:
    Detect targets from the given anime image.

    For example, you can detect the heads with :func:`imgutils.head.detect_heads` and visualize it
    with :func:`imgutils.visual.detection_visualize` like this

    .. image:: head_detect_demo.plot.py.svg
        :align: center
"""
from .censor import detect_censors
from .eye import detect_eyes
from .face import detect_faces
from .halfbody import detect_halfbody
from .hand import detect_hands
from .head import detect_heads
from .person import detect_person
from .text import detect_text
from .visual import detection_visualize
