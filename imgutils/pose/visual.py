import colorsys
from typing import List, Optional

import numpy as np
from PIL import ImageDraw, Image

from imgutils.data import ImageTyping, load_image
from .format import OP18KeyPointSet

_OP18_BODY_CONNECTS = [
    [2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
    [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
    [1, 16], [16, 18], [3, 17], [6, 18]
]
_OP18_BODY_CONNECTS = _OP18_BODY_CONNECTS[:17]
_OP18_BODY_COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0),
    (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255),
    (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85)
]


def _op18_body(keypoints: OP18KeyPointSet, draw: ImageDraw.ImageDraw, threshold: float = 0.3):
    for (ix, iy), color in zip(_OP18_BODY_CONNECTS, _OP18_BODY_COLORS):
        x0, y0, s0 = keypoints.all[ix - 1]
        x1, y1, s1 = keypoints.all[iy - 1]
        if s0 >= threshold and s1 >= threshold:
            draw.line([(x0, y0), (x1, y1)], width=5, fill=color)


_OP18_HAND_EDGES = [
    [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10],
    [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]
]


# noinspection DuplicatedCode
def _op18_hands(keypoints: OP18KeyPointSet, draw: ImageDraw.ImageDraw, threshold: float = 0.3):
    for hands in [keypoints.left_hand, keypoints.right_hand]:
        for ie, (e0, e1) in enumerate(_OP18_HAND_EDGES):
            x0, y0, s0 = hands[e0]
            x1, y1, s1 = hands[e1]
            if s0 >= threshold and s1 >= threshold:
                color = (np.array(colorsys.hsv_to_rgb(ie / float(len(_OP18_HAND_EDGES)), 1.0, 1.0)) * 255) \
                    .astype(np.uint8)
                draw.line([(x0, y0), (x1, y1)], width=3, fill=tuple(color.tolist()))


_OP18_FOOT_EDGES = [(2, 0), (2, 1)]


# noinspection DuplicatedCode
def _op18_feet(keypoints: OP18KeyPointSet, draw: ImageDraw.ImageDraw, threshold: float = 0.3):
    for hands in [keypoints.left_foot, keypoints.right_foot]:
        for ie, (e0, e1) in enumerate(_OP18_FOOT_EDGES):
            x0, y0, s0 = hands[e0]
            x1, y1, s1 = hands[e1]
            if s0 >= threshold and s1 >= threshold:
                color = (np.array(colorsys.hsv_to_rgb(ie / float(len(_OP18_FOOT_EDGES)), 1.0, 1.0)) * 255) \
                    .astype(np.uint8)
                draw.line([(x0, y0), (x1, y1)], width=3, fill=tuple(color.tolist()))


_FACE_POINT_SIZE = 4.5


def _op18_face(keypoints: OP18KeyPointSet, draw: ImageDraw.ImageDraw, threshold: float = 0.3):
    for x0, y0, s0 in keypoints.face:
        if s0 >= threshold:
            draw.ellipse(
                (
                    x0 - _FACE_POINT_SIZE / 2, y0 - _FACE_POINT_SIZE / 2,
                    x0 + _FACE_POINT_SIZE / 2, y0 + _FACE_POINT_SIZE / 2,
                ),
                fill='#00ff00', outline='#00ff00',
            )


def op18_visualize(image: ImageTyping, keypoints_list: List[OP18KeyPointSet], threshold: float = 0.3,
                   min_edge_size: Optional[int] = 512, draw_body: bool = True, draw_hands: bool = True,
                   draw_feet: bool = True, draw_face: bool = True) -> Image.Image:
    image = load_image(image, force_background='white', mode='RGB')
    if min_edge_size is not None and min(image.width, image.height) > min_edge_size:
        r = min(image.width, image.height) / min_edge_size
    else:
        r = None

    if r is not None:
        new_width = int(image.width / r)
        new_height = int(image.height / r)
        image = image.resize((new_width, new_height))

    draw = ImageDraw.Draw(image)
    for keypoints in keypoints_list:
        if r is not None:
            keypoints /= r
        if draw_body:
            _op18_body(keypoints, draw, threshold)
        if draw_hands:
            _op18_hands(keypoints, draw, threshold)
        if draw_feet:
            _op18_feet(keypoints, draw, threshold)
        if draw_face:
            _op18_face(keypoints, draw, threshold)

    return image
