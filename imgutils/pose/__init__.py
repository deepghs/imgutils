"""
Overview:
    Utilities to detect human keypoints in anime images.

    .. image:: pose_demo.plot.py.svg
        :align: center
"""
from .dwpose import dwpose_estimate
from .format import OP18KeyPointSet, OP18_BODY_MAX, OP18_BODY_MIN, OP18_FACE_MAX, OP18_FACE_MIN, \
    OP18_LEFT_FOOT_MAX, OP18_LEFT_FOOT_MIN, OP18_LEFT_HAND_MAX, OP18_LEFT_HAND_MIN, \
    OP18_RIGHT_FOOT_MAX, OP18_RIGHT_FOOT_MIN, OP18_RIGHT_HAND_MAX, OP18_RIGHT_HAND_MIN, OpenPose18
from .visual import op18_visualize
