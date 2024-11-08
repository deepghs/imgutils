"""
Overview:
    This module provides functionality for detecting nudity in images using the NudeNet model.
    
    The module includes functions for preprocessing images, running the NudeNet YOLO model,
    applying non-maximum suppression (NMS), and postprocessing the results. It utilizes
    ONNX models hosted on `deepghs/nudenet_onnx <https://huggingface.co/deepghs/nudenet_onnx>`_
    for efficient inference. The original project is
    `notAI-tech/NudeNet <https://github.com/notAI-tech/NudeNet>`_.
    
    .. collapse:: Overview of NudeNet Detect (NSFW Warning!!!)

        .. image:: nudenet_detect_demo.plot.py.svg
            :align: center
    
    The main function :func:`detect_with_nudenet` can be used to perform nudity detection on
    given images, returning a list of bounding boxes, labels, and confidence scores.
    
    This is an overall benchmark of all the nudenet models:

    .. image:: nudenet_detect_benchmark.plot.py.svg
        :align: center

    .. note::

        Here is a detailed list of labels from the NudeNet detection model and their respective meanings:

        .. list-table::
           :widths: 25 75
           :header-rows: 1

           * - Label
             - Description
           * - FEMALE_GENITALIA_COVERED
             - Detects covered female genitalia in the image.
           * - FACE_FEMALE
             - Detects the face of a female in the image.
           * - BUTTOCKS_EXPOSED
             - Detects exposed buttocks in the image.
           * - FEMALE_BREAST_EXPOSED
             - Detects exposed female breasts in the image.
           * - FEMALE_GENITALIA_EXPOSED
             - Detects exposed female genitalia in the image.
           * - MALE_BREAST_EXPOSED
             - Detects exposed male breasts in the image.
           * - ANUS_EXPOSED
             - Detects exposed anus in the image.
           * - FEET_EXPOSED
             - Detects exposed feet in the image.
           * - BELLY_COVERED
             - Detects a covered belly in the image.
           * - FEET_COVERED
             - Detects covered feet in the image.
           * - ARMPITS_COVERED
             - Detects covered armpits in the image.
           * - ARMPITS_EXPOSED
             - Detects exposed armpits in the image.
           * - FACE_MALE
             - Detects the face of a male in the image.
           * - BELLY_EXPOSED
             - Detects an exposed belly in the image.
           * - MALE_GENITALIA_EXPOSED
             - Detects exposed male genitalia in the image.
           * - ANUS_COVERED
             - Detects a covered anus in the image.
           * - FEMALE_BREAST_COVERED
             - Detects covered female breasts in the image.
           * - BUTTOCKS_COVERED
             - Detects covered buttocks in the image.


    .. note::
    
        This module requires onnxruntime version 1.18 or higher.
"""

from typing import Tuple, List

import numpy as np
from PIL import Image
from hbutils.testing.requires.version import VersionInfo
from huggingface_hub import hf_hub_download

from imgutils.data import ImageTyping
from imgutils.utils import open_onnx_model, ts_lru_cache
from ..data import load_image


def _check_compatibility() -> bool:
    """
    Check if the installed onnxruntime version is compatible with NudeNet.

    :raises EnvironmentError: If the onnxruntime version is less than 1.18.
    """
    import onnxruntime
    if VersionInfo(onnxruntime.__version__) < '1.18':
        raise EnvironmentError(f'Nudenet not supported on onnxruntime {onnxruntime.__version__}, '
                               f'please upgrade it to 1.18+ version.\n'
                               f'If you are running on CPU, use "pip install -U onnxruntime" .\n'
                               f'If you are running on GPU, use "pip install -U onnxruntime-gpu" .')  # pragma: no cover


_REPO_ID = 'deepghs/nudenet_onnx'


@ts_lru_cache()
def _open_nudenet_yolo():
    """
    Open and cache the NudeNet YOLO ONNX model.

    :return: The loaded ONNX model for YOLO.
    """
    return open_onnx_model(hf_hub_download(
        repo_id=_REPO_ID,
        repo_type='model',
        filename='320n.onnx',
    ))


@ts_lru_cache()
def _open_nudenet_nms():
    """
    Open and cache the NudeNet NMS ONNX model.

    :return: The loaded ONNX model for NMS.
    """
    return open_onnx_model(hf_hub_download(
        repo_id=_REPO_ID,
        repo_type='model',
        filename='nms-yolov8.onnx',
    ))


def _nn_preprocessing(image: ImageTyping, model_size: int = 320) -> Tuple[np.ndarray, float]:
    """
    Preprocess the input image for the NudeNet model.

    :param image: The input image.
    :param model_size: The size to which the image should be resized (default: 320).
    :return: A tuple containing the preprocessed image array and the scaling ratio.
    """
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
    """
    Create a configuration array for the NMS model.

    :param topk: The maximum number of detections to keep (default: 100).
    :param iou_threshold: The IoU threshold for NMS (default: 0.45).
    :param score_threshold: The score threshold for detections (default: 0.25).
    :return: A numpy array containing the configuration parameters.
    """
    return np.array([topk, iou_threshold, score_threshold]).astype(np.float32)


def _nn_postprocess(selected, global_ratio: float):
    """
    Postprocess the model output to generate bounding boxes and labels.

    :param selected: The output from the NMS model.
    :param global_ratio: The scaling ratio to apply to the bounding boxes.
    :return: A list of tuples, each containing a bounding box, label, and confidence score.
    """
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
    """
    Detect nudity in the given image using the NudeNet model.

    :param image: The input image to analyze.
    :param topk: The maximum number of detections to keep (default: 100).
    :param iou_threshold: The IoU threshold for NMS (default: 0.45).
    :param score_threshold: The score threshold for detections (default: 0.25).
    :return: A list of tuples, each containing:

             - A bounding box as (x1, y1, x2, y2)
             - A label string
             - A confidence score
    """
    _check_compatibility()
    input_, global_ratio = _nn_preprocessing(image, model_size=320)
    config = _make_np_config(topk, iou_threshold, score_threshold)
    output0, = _open_nudenet_yolo().run(['output0'], {'images': input_})
    selected, = _open_nudenet_nms().run(['selected'], {'detection': output0, 'config': config})
    return _nn_postprocess(selected[0], global_ratio=global_ratio)
