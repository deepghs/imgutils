from typing import List, Tuple

from .detect import _detect_text, _list_det_models
from .recognize import _text_recognize, _list_rec_models
from ..data import ImageTyping, load_image
from ..utils import tqdm

_DEFAULT_DET_MODEL = 'ch_PP-OCRv4_det'
_DEFAULT_REC_MODEL = 'ch_PP-OCRv4_rec'


def list_det_models() -> List[str]:
    return _list_det_models()


def list_rec_models() -> List[str]:
    return _list_rec_models()


def detect_text_with_ocr(image: ImageTyping, model: str = _DEFAULT_DET_MODEL,
                         heat_threshold: float = 0.3, box_threshold: float = 0.7,
                         max_candidates: int = 1000, unclip_ratio: float = 2.0) \
        -> List[Tuple[Tuple[int, int, int, int], str, float]]:
    retval = []
    for box, _, score in _detect_text(image, model, heat_threshold, box_threshold, max_candidates, unclip_ratio):
        retval.append((box, 'text', score))
    retval = sorted(retval, key=lambda x: x[2], reverse=True)
    return retval


def ocr(image: ImageTyping, detect_model: str = _DEFAULT_DET_MODEL,
        recognize_model: str = _DEFAULT_REC_MODEL, heat_threshold: float = 0.3, box_threshold: float = 0.7,
        max_candidates: int = 1000, unclip_ratio: float = 2.0, rotation_threshold: float = 1.5,
        is_remove_duplicate: bool = False, silent: bool = False):
    image = load_image(image)
    retval = []
    for (x0, y0, x1, y1), _, score in \
            tqdm(_detect_text(image, detect_model, heat_threshold,
                              box_threshold, max_candidates, unclip_ratio), silent=silent):
        width, height = x1 - x0, y1 - y0
        area = image.crop((x0, y0, x1, y1))
        if height >= width * rotation_threshold:
            area = area.rotate(90)

        text, rec_score = _text_recognize(area, recognize_model, is_remove_duplicate)
        retval.append(((x0, y0, x1, y1), text, score * rec_score))

    retval = sorted(retval, key=lambda x: x[2], reverse=True)
    return retval
