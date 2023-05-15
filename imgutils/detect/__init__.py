"""
Overview:
    Detect targets from the given anime image.

    For example, you can detect the faces with :func:`imgutils.face.detect_faces` and visualize it
    with :func:`imgutils.visual.detection_visualize` like this

    .. image:: face_detect.dat.svg
        :align: center
"""
from .face import detect_faces
from .person import detect_person
from .visual import detection_visualize
