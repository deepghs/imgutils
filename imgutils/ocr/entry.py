from typing import List, Tuple

from .detect import _detect_text
from .recognize import _text_recognize
from ..data import ImageTyping, load_image

_DEFAULT_MODEL = 'ch_PP-OCRv4_det_infer'


def detect_text_with_ocr(image: ImageTyping, model: str = _DEFAULT_MODEL,
                         heat_threshold: float = 0.3, box_threshold: float = 0.7,
                         max_candidates: int = 1000, unclip_ratio: float = 2.0) \
        -> List[Tuple[Tuple[int, int, int, int], str, float]]:
    retval = []
    for box, _, score in _detect_text(image, model, heat_threshold, box_threshold, max_candidates, unclip_ratio):
        retval.append((box, 'text', score))
    retval = sorted(retval, key=lambda x: x[2], reverse=True)
    return retval


def ocr(image: ImageTyping, model: str = _DEFAULT_MODEL,
        heat_threshold: float = 0.3, box_threshold: float = 0.7,
        max_candidates: int = 1000, unclip_ratio: float = 2.0,
        is_remove_duplicate: bool = False):
    image = load_image(image)
    retval = []
    for (x0, y0, x1, y1), _, score in _detect_text(image, model, heat_threshold,
                                                   box_threshold, max_candidates, unclip_ratio):
        width, height = x1 - x0, y1 - y0
        area = image.crop((x0, y0, x1, y1))
        if height >= width * 1.5:
            area = area.rotate(90)

        text, _ = _text_recognize(area, model, is_remove_duplicate)
        print(text, score)
        retval.append(((x0, y0, x1, y1), text, score))

    retval = sorted(retval, key=lambda x: x[2], reverse=True)
    return retval
