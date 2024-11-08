from typing import List

import cv2
import numpy as np
import pyclipper
from huggingface_hub import hf_hub_download, HfFileSystem
from shapely import Polygon

from ..data import ImageTyping, load_image
from ..utils import open_onnx_model, ts_lru_cache

_MIN_SIZE = 3
_HF_CLIENT = HfFileSystem()
_REPOSITORY = 'deepghs/paddleocr'


@ts_lru_cache()
def _open_ocr_detection_model(model):
    return open_onnx_model(hf_hub_download(
        _REPOSITORY,
        f'det/{model}/model.onnx',
    ))


def _box_score_fast(bitmap, _box):
    h, w = bitmap.shape[:2]
    box = _box.copy()
    xmin = np.clip(np.floor(box[:, 0].min()).astype("int32"), 0, w - 1)
    xmax = np.clip(np.ceil(box[:, 0].max()).astype("int32"), 0, w - 1)
    ymin = np.clip(np.floor(box[:, 1].min()).astype("int32"), 0, h - 1)
    ymax = np.clip(np.ceil(box[:, 1].max()).astype("int32"), 0, h - 1)

    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box[:, 0] = box[:, 0] - xmin
    box[:, 1] = box[:, 1] - ymin
    # noinspection PyTypeChecker
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype("int32"), 1)
    return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


def _unclip(box, unclip_ratio):
    poly = Polygon(box)
    distance = poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = np.array(offset.Execute(distance))
    return expanded


def _get_mini_boxes(contour):
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    if points[1][1] > points[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0

    if points[3][1] > points[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2

    box = [
        points[index_1], points[index_2], points[index_3], points[index_4]
    ]
    return box, min(bounding_box[1])


def _boxes_from_bitmap(pred, _bitmap, dest_width, dest_height,
                       box_threshold=0.7, max_candidates=1000, unclip_ratio=2.0):
    bitmap = _bitmap
    height, width = bitmap.shape

    outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(outs) == 3:
        img, contours, _ = outs[0], outs[1], outs[2]
    elif len(outs) == 2:
        contours, _ = outs[0], outs[1]

    # noinspection PyUnboundLocalVariable
    num_contours = min(len(contours), max_candidates)

    boxes = []
    scores = []
    for index in range(num_contours):
        contour = contours[index]
        points, sside = _get_mini_boxes(contour)
        if sside < _MIN_SIZE:
            continue
        points = np.array(points)
        score = _box_score_fast(pred, points.reshape(-1, 2))
        if box_threshold > score:
            continue

        box = _unclip(points, unclip_ratio).reshape(-1, 1, 2)
        box, sside = _get_mini_boxes(box)
        if sside < _MIN_SIZE + 2:
            continue
        box = np.array(box)

        box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
        box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
        boxes.append(box.astype("int32"))
        scores.append(score)
    return np.array(boxes, dtype="int32"), scores


def _normalize(data, mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)):
    mean, std = np.asarray(mean), np.asarray(std)
    return (data - mean[None, :, None, None]) / std[None, :, None, None]


_ALIGN = 64


def _get_text_points(image: ImageTyping, model: str = 'ch_PP-OCRv4_det',
                     heat_threshold: float = 0.3, box_threshold: float = 0.7,
                     max_candidates: int = 1000, unclip_ratio: float = 2.0):
    origin_width, origin_height = width, height = image.size
    if width % _ALIGN != 0:
        width += (_ALIGN - width % _ALIGN)
    if height % _ALIGN != 0:
        height += (_ALIGN - height % _ALIGN)

    input_ = np.array(image).transpose((2, 0, 1)).astype(np.float32) / 255.0
    # noinspection PyTypeChecker
    input_ = np.pad(input_[None, ...], ((0, 0), (0, 0), (0, height - origin_height), (0, width - origin_width)))

    _ort_session = _open_ocr_detection_model(model)

    input_ = _normalize(input_).astype(np.float32)
    _input_name = _ort_session.get_inputs()[0].name
    _output_name = _ort_session.get_outputs()[0].name
    output_, = _ort_session.run([_output_name], {_input_name: input_})
    heatmap = output_[0][0]
    heatmap = heatmap[:origin_height, :origin_width]

    retval = []
    for points, score in zip(*_boxes_from_bitmap(
            heatmap, heatmap >= heat_threshold, origin_width, origin_height,
            box_threshold, max_candidates, unclip_ratio,
    )):
        retval.append((points, score))
    return retval


def _detect_text(image: ImageTyping, model: str = 'ch_PP-OCRv4_det',
                 heat_threshold: float = 0.3, box_threshold: float = 0.7,
                 max_candidates: int = 1000, unclip_ratio: float = 2.0):
    image = load_image(image, force_background='white', mode='RGB')
    retval = []
    for points, score in _get_text_points(image, model, heat_threshold, box_threshold, max_candidates, unclip_ratio):
        x0, y0 = points[:, 0].min(), points[:, 1].min()
        x1, y1 = points[:, 0].max(), points[:, 1].max()
        retval.append(((x0.item(), y0.item(), x1.item(), y1.item()), 'text', score))

    return retval


@ts_lru_cache()
def _list_det_models() -> List[str]:
    retval = []
    repo_segment_cnt = len(_REPOSITORY.split('/'))
    for item in _HF_CLIENT.glob(f'{_REPOSITORY}/det/*/model.onnx'):
        retval.append(item.split('/')[repo_segment_cnt:][1])
    return retval
