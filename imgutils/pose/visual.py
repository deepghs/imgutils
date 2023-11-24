from typing import List

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


def op18_visualize(image: ImageTyping, keypoints_list: List[OP18KeyPointSet], threshold: float = 0.3,
                   draw_body: bool = True) -> Image.Image:
    image = load_image(image, force_background='white', mode='RGB')
    draw = ImageDraw.Draw(image)
    for keypoints in keypoints_list:
        if draw_body:
            _op18_body(keypoints, draw, threshold)

    return image
