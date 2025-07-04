"""
Overview:
    This module provides functionality for visualizing object detection results on images.
    It includes tools for drawing bounding boxes, labels, and confidence scores on detected objects.

    The main function :func:`detection_visualize` can be used to visualize detection results from
    various object detection models, with customizable appearance settings like font size, padding,
    and label visibility.

    See :func:`imgutils.detect.head.detect_heads` and :func:`imgutils.detect.person.detect_person` for examples.
"""
import math
from typing import List, Optional, Union

import numpy as np
from PIL import ImageFont, ImageDraw, Image
from hbutils.color import rnd_colors, Color

from .base import BBoxWithScoreAndLabel, MaskWithScoreAndLabel
from ..data import ImageTyping, load_image


def _try_get_font_from_matplotlib(fp=None, fontsize: int = 12):
    """
    Attempt to get a font from matplotlib for text rendering.

    :param fp: Font properties object or None. If None, uses default sans-serif font.
    :type fp: matplotlib.font_manager.FontProperties or None
    :param fontsize: Size of the font to be used.
    :type fontsize: int

    :return: A PIL ImageFont object if matplotlib is available, None otherwise.
    :rtype: PIL.ImageFont.FreeTypeFont or None
    """
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


def detection_visualize(image: ImageTyping, detection: List[Union[BBoxWithScoreAndLabel, MaskWithScoreAndLabel]],
                        labels: Optional[List[str]] = None, text_padding: int = 6, fontsize: int = 12,
                        max_short_edge_size: Optional[int] = None, mask_alpha: float = 0.5,
                        fp=None, no_label: bool = False):
    """
    Visualize object detection results by drawing bounding boxes, masks, and labels on an image.

    This function takes detection results (bounding boxes and/or masks) and renders them on the input image,
    with customizable appearance settings. It supports both bounding box and instance segmentation results.

    :param image: Input image to visualize detections on. Can be a PIL Image, numpy array, or path to image file.
    :type image: ImageTyping
    :param detection: List of detection results, each containing bounding box coordinates, label, confidence score,
                      and optionally a segmentation mask. The coordinates should be in pixels, not normalized.
    :type detection: List[Union[BBoxWithScoreAndLabel, MaskWithScoreAndLabel]]
    :param labels: List of predefined labels. If None, labels will be extracted from detection results.
    :type labels: Optional[List[str]]
    :param text_padding: Padding around label text in pixels.
    :type text_padding: int
    :param fontsize: Font size for label text.
    :type fontsize: int
    :param max_short_edge_size: Maximum size of shortest image edge. If specified, image will be resized
                                while maintaining aspect ratio.
    :type max_short_edge_size: Optional[int]
    :param mask_alpha: Transparency level for mask visualization (0.0 to 1.0).
    :type mask_alpha: float
    :param fp: Font properties for matplotlib font. Only used if matplotlib is available.
    :type fp: matplotlib.font_manager.FontProperties or None
    :param no_label: If True, suppresses drawing of labels.
    :type no_label: bool

    :return: PIL Image with visualized detection results.
    :rtype: PIL.Image.Image

    Examples::
        >>> from imgutils.detect import detect_heads, detection_visualize
        >>> from imgutils.data import load_image
        >>>
        >>> # Basic usage
        >>> image = load_image("path/to/image.jpg")
        >>> detections = detect_heads(image)
        >>> visualized = detection_visualize(image, detections)
        >>> visualized.save("output.png")

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

    labels = sorted(labels or {label for _, label, *_ in detection})
    _colors = list(map(str, rnd_colors(len(labels))))
    _color_map = dict(zip(labels, _colors))
    for _, detect_item in sorted(enumerate(detection), key=lambda x: (x[1][2], x[0])):
        if len(detect_item) == 3:
            (x0, y0, x1, y1), label, score = detect_item
            mask = None
        else:
            (x0, y0, x1, y1), label, score, mask = detect_item
        x0, y0 = int(x0 * new_width / original_width), int(y0 * new_height / original_height)
        x1, y1 = int(x1 * new_width / original_width), int(y1 * new_height / original_height)
        box_color = _color_map[label]
        draw.rectangle((x0, y0, x1, y1), outline=box_color, width=2)

        if mask is not None:
            if mask.shape[0] != new_height or mask.shape[1] != new_width:
                mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
                mask_pil = mask_pil.resize((new_width, new_height), Image.BILINEAR)
                mask = np.array(mask_pil) / 255.0

            color = Color(box_color)
            overlay = np.zeros((new_height, new_width, 4), dtype=np.uint8)
            overlay[..., 0] = int(color.rgb.red * 255)
            overlay[..., 1] = int(color.rgb.green * 255)
            overlay[..., 2] = int(color.rgb.blue * 255)
            overlay[..., 3] = (mask * mask_alpha * 255).astype(np.uint8)

            overlay_pil = Image.fromarray(overlay)
            visual_image.paste(overlay_pil, (0, 0), overlay_pil)

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
