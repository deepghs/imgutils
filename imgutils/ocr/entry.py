from typing import List, Tuple

from .detect import _detect_text, _list_det_models
from .recognize import _text_recognize, _list_rec_models
from ..data import ImageTyping, load_image

_DEFAULT_DET_MODEL = 'ch_PP-OCRv4_det'
_DEFAULT_REC_MODEL = 'ch_PP-OCRv4_rec'


def list_det_models() -> List[str]:
    """
    List available text detection models for OCR.

    :return: A list of available text detection model names.
    :rtype: List[str]
    """
    return _list_det_models()


def list_rec_models() -> List[str]:
    """
    List available text recognition models for OCR.

    :return: A list of available text recognition model names.
    :rtype: List[str]
    """
    return _list_rec_models()


def detect_text_with_ocr(image: ImageTyping, model: str = _DEFAULT_DET_MODEL,
                         heat_threshold: float = 0.3, box_threshold: float = 0.7,
                         max_candidates: int = 1000, unclip_ratio: float = 2.0) \
        -> List[Tuple[Tuple[int, int, int, int], str, float]]:
    """
    Detect text in an image using an OCR model.

    :param image: The input image.
    :type image: ImageTyping
    :param model: The name of the text detection model.
    :type model: str, optional
    :param heat_threshold: The heat map threshold for text detection.
    :type heat_threshold: float, optional
    :param box_threshold: The box threshold for text detection.
    :type box_threshold: float, optional
    :param max_candidates: The maximum number of candidates to consider.
    :type max_candidates: int, optional
    :param unclip_ratio: The unclip ratio for text detection.
    :type unclip_ratio: float, optional
    :return: A list of detected text boxes, their corresponding text content, and their confidence scores.
    :rtype: List[Tuple[Tuple[int, int, int, int], str, float]]
    """
    retval = []
    for box, _, score in _detect_text(image, model, heat_threshold, box_threshold, max_candidates, unclip_ratio):
        retval.append((box, 'text', score))
    retval = sorted(retval, key=lambda x: x[2], reverse=True)
    return retval


def ocr(image: ImageTyping, detect_model: str = _DEFAULT_DET_MODEL,
        recognize_model: str = _DEFAULT_REC_MODEL, heat_threshold: float = 0.3, box_threshold: float = 0.7,
        max_candidates: int = 1000, unclip_ratio: float = 2.0, rotation_threshold: float = 1.5,
        is_remove_duplicate: bool = False):
    """
    Perform optical character recognition (OCR) on an image.

    :param image: The input image.
    :type image: ImageTyping
    :param detect_model: The name of the text detection model.
    :type detect_model: str, optional
    :param recognize_model: The name of the text recognition model.
    :type recognize_model: str, optional
    :param heat_threshold: The heat map threshold for text detection.
    :type heat_threshold: float, optional
    :param box_threshold: The box threshold for text detection.
    :type box_threshold: float, optional
    :param max_candidates: The maximum number of candidates to consider.
    :type max_candidates: int, optional
    :param unclip_ratio: The unclip ratio for text detection.
    :type unclip_ratio: float, optional
    :param rotation_threshold: The rotation threshold for text detection.
    :type rotation_threshold: float, optional
    :param is_remove_duplicate: Whether to remove duplicate text content.
    :type is_remove_duplicate: bool, optional
    :return: A list of detected text boxes, their corresponding text content, and their combined confidence scores.
    :rtype: List[Tuple[Tuple[int, int, int, int], str, float]]
    """
    image = load_image(image)
    retval = []
    for (x0, y0, x1, y1), _, score in \
            _detect_text(image, detect_model, heat_threshold, box_threshold, max_candidates, unclip_ratio):
        width, height = x1 - x0, y1 - y0
        area = image.crop((x0, y0, x1, y1))
        if height >= width * rotation_threshold:
            area = area.rotate(90)

        text, rec_score = _text_recognize(area, recognize_model, is_remove_duplicate)
        retval.append(((x0, y0, x1, y1), text, score * rec_score))

    retval = sorted(retval, key=lambda x: x[2], reverse=True)
    return retval
