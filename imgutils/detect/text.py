"""
Overview:
    Detect text in images.

    Models are hosted on `deepghs/text_detection <https://huggingface.co/spaces/deepghs/text_detection>`_.

    .. image:: text_detect_demo.plot.py.svg
        :align: center

    This is an overall benchmark of all the text detect models:

    .. image:: text_detect_benchmark.plot.py.svg
        :align: center

    .. warning::
        This module has been deprecated and will be removed in the future.

        It is recommended to migrate to the :func:`imgutils.ocr.detect_text_with_ocr` function as soon as possible.
        This function uses a higher-quality text detection model provided by PaddleOCR,
        resulting in improved performance and higher efficiency.

        .. image:: text_detect_deprecate_demo.plot.py.svg
            :align: center

"""
from typing import List, Tuple, Optional

import cv2
import numpy as np
from deprecation import deprecated
from huggingface_hub import hf_hub_download

from ..config.meta import __VERSION__
from ..data import ImageTyping, load_image
from ..utils import open_onnx_model, ts_lru_cache

_DEFAULT_MODEL = 'dbnetpp_resnet50_fpnc_1200e_icdar2015'


@ts_lru_cache()
def _open_text_detect_model(model: str):
    """
    Get an ONNX session for the specified DBNET or DBNET++ model.

    This function downloads the ONNX model and opens it using the imgutils library.

    :param model: Model name for DBNET or DBNET++.
    :type model: str
    :return: ONNX session for the specified model.
    """
    return open_onnx_model(hf_hub_download(
        'deepghs/text_detection',
        f'{model}/end2end.onnx'
    ))


def _get_heatmap_of_text(image: ImageTyping, model: str) -> np.ndarray:
    """
    Get the heatmap of text regions from the given image using the specified model.

    :param image: Input image.
    :type image: ImageTyping
    :param model: Model name for DBNET or DBNET++.
    :type model: str
    :return: Heatmap of text regions.
    :rtype: np.ndarray
    """
    origin_width, origin_height = width, height = image.size
    align = 32
    if width % align != 0:
        width += (align - width % align)
    if height % align != 0:
        height += (align - height % align)

    input_ = np.array(image).transpose((2, 0, 1)).astype(np.float32) / 255.0
    input_ = np.pad(input_[None, ...], ((0, 0), (0, 0), (0, height - origin_height), (0, width - origin_width)))

    def _normalize(data, mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)):
        mean, std = np.asarray(mean), np.asarray(std)
        return (data - mean[None, :, None, None]) / std[None, :, None, None]

    ort = _open_text_detect_model(model)

    input_ = _normalize(input_).astype(np.float32)
    output_, = ort.run(['output'], {'input': input_})
    heatmap = output_[0]
    heatmap = heatmap[:origin_height, :origin_width]

    return heatmap


def _get_bounding_box_of_text(image: ImageTyping, model: str, threshold: float) \
        -> List[Tuple[Tuple[int, int, int, int], float]]:
    """
    Get bounding boxes of detected text regions from the given image using the specified model and threshold.

    :param image: Input image.
    :type image: ImageTyping
    :param model: Model name for DBNET or DBNET++.
    :type model: str
    :param threshold: Confidence threshold for text detection.
    :type threshold: float
    :return: List of bounding boxes and their scores.
    :rtype: List[Tuple[Tuple[int, int, int, int], float]]
    """
    heatmap = _get_heatmap_of_text(image, model)
    c_rets = cv2.findContours((heatmap * 255.0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = c_rets[0] if len(c_rets) == 2 else c_rets[1]
    bboxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        x0, y0, x1, y1 = x, y, x + w, y + h
        score = heatmap[y0:y1, x0:x1].mean().item()
        if score >= threshold:
            bboxes.append(((x0, y0, x1, y1), score))

    return bboxes


@deprecated(deprecated_in="0.2.10", current_version=__VERSION__,
            details="Use the new function :func:`imgutils.ocr.detect_text_with_ocr` instead")
def detect_text(image: ImageTyping, model: str = _DEFAULT_MODEL, threshold: float = 0.05,
                max_area_size: Optional[int] = 640):
    """
    Detect text regions in the given image using the specified model and threshold.

    :param image: Input image.
    :type image: ImageTyping
    :param model: Model name for DBNET or DBNET++.
    :type model: str
    :param threshold: Confidence threshold for text detection.
    :type threshold: float
    :param max_area_size: Max area size when doing inference. Default is ``640``, which means if
        the image's area is over 640x640, it will be resized. When assigned to ``None``,
        it means do not resize in any case.
    :type max_area_size: Optional[int]
    :return: List of detected text bounding boxes, labels, and scores.
    :rtype: List[Tuple[Tuple[int, int, int, int], str, float]]

    .. warning::
        This function is deprecated, and it will be removed from imgutils in the future.
        Please migrate to :func:`imgutils.ocr.detect_text_with_ocr` as soon as possible.
    """
    image = load_image(image, mode='RGB')
    if max_area_size is not None and image.width * image.height >= max_area_size ** 2:
        r = ((image.width * image.height) / (max_area_size ** 2)) ** 0.5
        new_width, new_height = int(image.width / r), int(image.height / r)
        image = image.resize((new_width, new_height))
    else:
        r = 1.0

    bboxes = []
    for (x0, y0, x1, y1), score in _get_bounding_box_of_text(image, model, threshold):
        x0, y0, x1, y1 = int(x0 * r), int(y0 * r), int(x1 * r), int(y1 * r)
        bboxes.append(((x0, y0, x1, y1), 'text', score))

    bboxes = sorted(bboxes, key=lambda x: x[2], reverse=True)
    return bboxes
