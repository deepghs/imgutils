# NudeNet Model, from https://github.com/notAI-tech/NudeNet
# The ONNX models are hosted on https://huggingface.co/deepghs/nudenet_onnx
from functools import lru_cache
from typing import Tuple, List

import numpy as np
from PIL import Image
from hbutils.testing.requires.version import VersionInfo
from huggingface_hub import hf_hub_download

from imgutils.data import ImageTyping
from imgutils.utils import open_onnx_model
from ..data import load_image


def _check_compatibility() -> bool:
    import onnxruntime
    if VersionInfo(onnxruntime.__version__) < '1.18':
        raise EnvironmentError(f'Nudenet not supported on onnxruntime {onnxruntime.__version__}, '
                               f'please upgrade it to 1.18+ version.\n'
                               f'If you are running on CPU, use "pip install -U onnxruntime" .\n'
                               f'If you are running on GPU, use "pip install -U onnxruntime-gpu" .')  # pragma: no cover


_REPO_ID = 'deepghs/nudenet_onnx'


@lru_cache()
def _open_nudenet_yolo():
    return open_onnx_model(hf_hub_download(
        repo_id=_REPO_ID,
        repo_type='model',
        filename='320n.onnx',
    ))


@lru_cache()
def _open_nudenet_nms():
    return open_onnx_model(hf_hub_download(
        repo_id=_REPO_ID,
        repo_type='model',
        filename='nms-yolov8.onnx',
    ))


def _nn_preprocessing(image: ImageTyping, model_size: int = 320) \
        -> Tuple[np.ndarray, float]:
    image = load_image(image, mode='RGB', force_background='white')
    assert image.mode == 'RGB'
    mat = np.array(image)

    max_size = max(image.width, image.height)

    mat_pad = np.zeros((max_size, max_size, 3), dtype=np.uint8)
    mat_pad[:mat.shape[0], :mat.shape[1], :] = mat
    img_resized = Image.fromarray(mat_pad, mode='RGB').resize((model_size, model_size), resample=Image.BILINEAR)

    input_data = np.array(img_resized).transpose(2, 0, 1).astype(np.float32) / 255.0
    input_data = np.expand_dims(input_data, axis=0)
    return input_data, max_size / model_size


def _make_np_config(topk: int = 100, iou_threshold: float = 0.45, score_threshold: float = 0.25) -> np.ndarray:
    return np.array([topk, iou_threshold, score_threshold]).astype(np.float32)


def _nn_postprocess(selected, global_ratio: float):
    bboxes = []
    num_boxes = selected.shape[0]
    for idx in range(num_boxes):
        data = selected[idx, :]

        scores = data[4:]
        score = np.max(scores)
        label = np.argmax(scores)

        box = data[:4] * global_ratio
        x = (box[0] - 0.5 * box[2]).item()
        y = (box[1] - 0.5 * box[3]).item()
        w = box[2].item()
        h = box[3].item()

        bboxes.append(((x, y, x + w, y + h), _LABELS[label], score.item()))

    return bboxes


_LABELS = [
    "FEMALE_GENITALIA_COVERED",
    "FACE_FEMALE",
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "ANUS_EXPOSED",
    "FEET_EXPOSED",
    "BELLY_COVERED",
    "FEET_COVERED",
    "ARMPITS_COVERED",
    "ARMPITS_EXPOSED",
    "FACE_MALE",
    "BELLY_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_COVERED",
    "FEMALE_BREAST_COVERED",
    "BUTTOCKS_COVERED"
]


def detect_with_nudenet(image: ImageTyping, topk: int = 100,
                        iou_threshold: float = 0.45, score_threshold: float = 0.25) \
        -> List[Tuple[Tuple[int, int, int, int], str, float]]:
    _check_compatibility()
    input_, global_ratio = _nn_preprocessing(image, model_size=320)
    config = _make_np_config(topk, iou_threshold, score_threshold)
    output0, = _open_nudenet_yolo().run(['output0'], {'images': input_})
    selected, = _open_nudenet_nms().run(['selected'], {'detection': output0, 'config': config})
    return _nn_postprocess(selected[0], global_ratio=global_ratio)
