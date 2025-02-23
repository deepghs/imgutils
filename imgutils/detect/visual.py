"""
Overview:
    Visualize the detection results.

    See :func:`imgutils.detect.head.detect_heads` and :func:`imgutils.detect.person.detect_person` for examples.
"""
import math
from typing import List, Tuple, Optional

from PIL import ImageFont, ImageDraw
from hbutils.color import rnd_colors, Color

from imgutils.data import ImageTyping, load_image


def _try_get_font_from_matplotlib(fp=None, fontsize: int = 12):
    try:
        # noinspection PyPackageRequirements
        import matplotlib
    except (ModuleNotFoundError, ImportError):
        return None
    else:
        # noinspection PyPackageRequirements
        from matplotlib.font_manager import findfont, FontProperties
        font = findfont(fp or FontProperties(family=['sans-serif']))
        return ImageFont.truetype(font, fontsize)


def detection_visualize(image: ImageTyping, detection: List[Tuple[Tuple[float, float, float, float], str, float]],
                        labels: Optional[List[str]] = None, text_padding: int = 6, fontsize: int = 12,
                        max_short_edge_size: Optional[int] = None, fp=None, no_label: bool = False):
    """
    Overview:
        Visualize the results of the object detection.

    :param image: Image be detected.
    :param detection: The detection results list, each item includes the detected area `(x0, y0, x1, y1)`,
        the target type (always `head`) and the target confidence score.
    :param labels: An array of known labels. If not provided, the labels will be automatically detected
        from the given ``detection``.
    :param text_padding: Text padding of the labels. Default is ``6``.
    :param fontsize: Font size of the labels. At runtime, an attempt will be made to retrieve the font used
        for rendering from `matplotlib`. Therefore, if `matplotlib` is not installed, only the default pixel font
        provided with `Pillow` can be used, and the font size cannot be changed.
    :param no_label: Do not show labels. Default is ``False``.
    :return: A `PIL` image with the same size as the provided image `image`, which contains the original image
        content as well as the visualized bounding boxes.

    Examples::
        See :func:`imgutils.detect.head.detect_heads` and :func:`imgutils.detect.person.detect_person` for examples.
    """
    image = load_image(image, force_background=None, mode='RGBA')
    original_width, original_height = image.width, image.height
    if max_short_edge_size is not None and max_short_edge_size < min(original_height, original_width):
        r = max_short_edge_size / min(original_height, original_width)
        new_width = int(math.ceil(original_width * r))
        new_height = int(math.ceil(original_height * r))
    else:
        new_width, new_height = original_width, original_height

    visual_image = image.copy()
    if (new_width, new_height) != (original_width, original_height):
        visual_image = visual_image.resize((new_width, new_height))
    draw = ImageDraw.Draw(visual_image, mode='RGBA')
    font = _try_get_font_from_matplotlib(fp, fontsize) or ImageFont.load_default()

    labels = sorted(labels or {label for _, label, _ in detection})
    _colors = list(map(str, rnd_colors(len(labels))))
    _color_map = dict(zip(labels, _colors))
    for _, ((x0, y0, x1, y1), label, score) in sorted(enumerate(detection), key=lambda x: (x[1][2], x[0])):
        x0, y0 = int(x0 * new_width / original_width), int(y0 * new_height / original_height)
        x1, y1 = int(x1 * new_width / original_width), int(y1 * new_height / original_height)
        box_color = _color_map[label]
        draw.rectangle((x0, y0, x1, y1), outline=box_color, width=2)

        if not no_label:
            label_text = f'{label}: {score * 100:.2f}%'
            _t_x0, _t_y0, _t_x1, _t_y1 = draw.textbbox((x0, y0), label_text, font=font)
            _t_width, _t_height = _t_x1 - _t_x0, _t_y1 - _t_y0
            if y0 - _t_height - text_padding < 0:
                _t_text_rect = (x0, y0, x0 + _t_width + text_padding * 2, y0 + _t_height + text_padding * 2)
                _t_text_co = (x0 + text_padding, y0 + text_padding)
            else:
                _t_text_rect = (x0, y0 - _t_height - text_padding * 2, x0 + _t_width + text_padding * 2, y0)
                _t_text_co = (x0 + text_padding, y0 - _t_height - text_padding)

            draw.rectangle(_t_text_rect, fill=str(Color(box_color, alpha=0.5)))
            draw.text(_t_text_co, label_text, fill="black", font=font)

    return visual_image
