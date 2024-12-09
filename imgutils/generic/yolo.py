"""
This module provides functionality for YOLO object detection using ONNX models from Hugging Face.

It includes utilities for preprocessing images, performing object detection, and post-processing
the results. The main components are:

1. YOLOModel class: Manages YOLO models from a Hugging Face repository.
2. Helper functions for coordinate conversion, non-maximum suppression, and image processing.
3. A high-level function 'yolo_predict' for easy object detection on images.

The module supports various image input types and allows customization of confidence and IoU thresholds.
"""

import ast
import json
import math
import os
from threading import Lock
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image
from hbutils.color import rnd_colors
from hfutils.operate import get_hf_client, get_hf_fs
from hfutils.repository import hf_hub_repo_url
from hfutils.utils import hf_fs_path, hf_normpath
from huggingface_hub import HfFileSystem, hf_hub_download

from ..data import load_image, rgb_encode, ImageTyping
from ..utils import open_onnx_model, ts_lru_cache

try:
    import gradio as gr
except (ImportError, ModuleNotFoundError):
    gr = None

__all__ = [
    'YOLOModel',
    'yolo_predict',
]


def _check_gradio_env():
    """
    Check if the Gradio library is installed and available.

    :raises EnvironmentError: If Gradio is not installed.
    """
    if gr is None:
        raise EnvironmentError(f'Gradio required for launching webui-based demo.\n'
                               f'Please install it with `pip install dghs-imgutils[demo]`.')


def _v_fix(v):
    """
    Round and convert a float value to an integer.

    :param v: The float value to be rounded and converted.
    :type v: float
    :return: The rounded integer value.
    :rtype: int
    """
    return int(round(v))


def _bbox_fix(bbox):
    """
    Fix the bounding box coordinates by rounding them to integers.

    :param bbox: The bounding box coordinates.
    :type bbox: tuple
    :return: A tuple of fixed (rounded to integer) bounding box coordinates.
    :rtype: tuple
    """
    return tuple(map(_v_fix, bbox))


def _yolo_xywh2xyxy(x: np.ndarray) -> np.ndarray:
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format.

    This function is adapted from YOLOv8 and transforms the center-based representation
    to a corner-based representation of bounding boxes.

    :param x: Input bounding box coordinates in (x, y, width, height) format.
    :type x: np.ndarray

    :return: Bounding box coordinates in (x1, y1, x2, y2) format.
    :rtype: np.ndarray

    :Example:

    >>> import numpy as np
    >>> boxes = np.array([[10, 10, 20, 20]])
    >>> _yolo_xywh2xyxy(boxes)
    array([[  0.,   0.,  20.,  20.]])
    """
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def _yolo_nms(boxes, scores, iou_threshold: float = 0.7) -> List[int]:
    """
    Perform Non-Maximum Suppression (NMS) on bounding boxes.

    This function applies NMS to remove overlapping bounding boxes, keeping only the most confident detections.

    :param boxes: Array of bounding boxes, each in the format [xmin, ymin, xmax, ymax].
    :type boxes: np.ndarray
    :param scores: Array of confidence scores for each bounding box.
    :type scores: np.ndarray
    :param iou_threshold: IoU threshold for considering boxes as overlapping. Default is 0.7.
    :type iou_threshold: float

    :return: List of indices of the boxes to keep after NMS.
    :rtype: List[int]

    :Example:

    >>> boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30]])
    >>> scores = np.array([0.9, 0.8, 0.7])
    >>> _yolo_nms(boxes, scores, 0.5)
    [0, 2]
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)

        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


def _image_preprocess(image: Image.Image, max_infer_size: Union[int, Tuple[int, int]] = 1216,
                      allow_dynamic: bool = False, align: int = 32):
    """
    Preprocess an input image for inference.

    This function resizes the image while maintaining its aspect ratio, and ensures
    the dimensions are multiples of 'align'.

    :param image: Input image to be preprocessed.
    :type image: Image.Image
    :param max_infer_size: Maximum size (width or height) of the processed image. Default is 1216.
    :type max_infer_size: int
    :param allow_dynamic: If True, allows dynamic resizing of the image while maintaining aspect ratio. Default is False.
    :type allow_dynamic: bool
    :param align: Value to align the image dimensions to. Default is 32.
    :type align: int

    :return: A tuple containing:
        - The preprocessed image
        - Original image dimensions (width, height)
        - New image dimensions (width, height)
    :rtype: tuple(Image.Image, Tuple[int, int], Tuple[int, int])

    :Example:

    >>> from PIL import Image
    >>> img = Image.new('RGB', (1000, 800))
    >>> processed_img, old_size, new_size = _image_preprocess(img)
    >>> print(old_size, new_size)
    (1000, 800) (1216, 992)
    """
    if isinstance(max_infer_size, int):
        max_infer_width, max_infer_height = max_infer_size, max_infer_size
    else:
        max_infer_width, max_infer_height = max_infer_size

    old_width, old_height = image.width, image.height
    new_width, new_height = old_width, old_height
    if allow_dynamic:
        r = min(max_infer_width / new_width, max_infer_height / new_height)
        if r < 1:
            new_width, new_height = new_width * r, new_height * r
        new_width = int(math.ceil(new_width / align) * align)
        new_height = int(math.ceil(new_height / align) * align)
    else:
        new_width, new_height = max_infer_width, max_infer_height

    image = image.resize((new_width, new_height))
    return image, (old_width, old_height), (new_width, new_height)


def _xy_postprocess(x, y, old_size: Tuple[float, float], new_size: Tuple[float, float]):
    """
    Convert coordinates from the preprocessed image size back to the original image size.

    :param x: X-coordinate in the preprocessed image.
    :type x: float
    :param y: Y-coordinate in the preprocessed image.
    :type y: float
    :param old_size: Original image dimensions (width, height).
    :type old_size: Tuple[float, float]
    :param new_size: Preprocessed image dimensions (width, height).
    :type new_size: Tuple[float, float]

    :return: Adjusted (x, y) coordinates for the original image size.
    :rtype: Tuple[int, int]

    :Example:

    >>> _xy_postprocess(100, 100, (1000, 800), (1216, 992))
    (82, 80)
    """
    old_width, old_height = old_size
    new_width, new_height = new_size
    x, y = x / new_width * old_width, y / new_height * old_height
    x = int(np.clip(x, a_min=0, a_max=old_width).round())
    y = int(np.clip(y, a_min=0, a_max=old_height).round())
    return x, y


def _end2end_postprocess(output, conf_threshold: float, iou_threshold: float,
                         old_size: Tuple[float, float], new_size: Tuple[float, float], labels: List[str]) \
        -> List[Tuple[Tuple[int, int, int, int], str, float]]:
    """
    Post-process the output of an end-to-end object detection model.

    This function filters detections based on confidence, applies non-maximum suppression,
    and transforms coordinates back to the original image size.

    :param output: Raw output from the end-to-end object detection model.
    :type output: np.ndarray
    :param conf_threshold: Confidence threshold for filtering detections.
    :type conf_threshold: float
    :param iou_threshold: IoU threshold for non-maximum suppression (not used in this function).
    :type iou_threshold: float
    :param old_size: Original image dimensions (width, height).
    :type old_size: Tuple[float, float]
    :param new_size: Preprocessed image dimensions (width, height).
    :type new_size: Tuple[float, float]
    :param labels: List of class labels.
    :type labels: List[str]

    :return: List of detections, each in the format ((x0, y0, x1, y1), label, confidence).
    :rtype: List[Tuple[Tuple[int, int, int, int], str, float]]

    :raises AssertionError: If the output shape is not as expected.
    """
    assert output.shape[-1] == 6
    _ = iou_threshold  # actually the iou_threshold has not been supplied to end2end post-processing
    detections = []
    output = output[output[:, 4] > conf_threshold]
    selected_idx = _yolo_nms(output[:, :4], output[:, 4])
    for x0, y0, x1, y1, score, cls in output[selected_idx]:
        x0, y0 = _xy_postprocess(x0, y0, old_size, new_size)
        x1, y1 = _xy_postprocess(x1, y1, old_size, new_size)
        detections.append(((x0, y0, x1, y1), labels[int(cls.item())], float(score)))

    return detections


def _nms_postprocess(output, conf_threshold: float, iou_threshold: float,
                     old_size: Tuple[float, float], new_size: Tuple[float, float], labels: List[str]) \
        -> List[Tuple[Tuple[int, int, int, int], str, float]]:
    """
    Post-process the output of an NMS-based object detection model.

    This function applies confidence thresholding, non-maximum suppression,
    and transforms coordinates back to the original image size.

    :param output: Raw output from the NMS-based object detection model.
    :type output: np.ndarray
    :param conf_threshold: Confidence threshold for filtering detections.
    :type conf_threshold: float
    :param iou_threshold: IoU threshold for non-maximum suppression.
    :type iou_threshold: float
    :param old_size: Original image dimensions (width, height).
    :type old_size: Tuple[float, float]
    :param new_size: Preprocessed image dimensions (width, height).
    :type new_size: Tuple[float, float]
    :param labels: List of class labels.
    :type labels: List[str]

    :return: List of detections, each in the format ((x0, y0, x1, y1), label, confidence).
    :rtype: List[Tuple[Tuple[int, int, int, int], str, float]]

    :raises AssertionError: If the output shape is not as expected.
    """
    assert output.shape[0] == 4 + len(labels)
    # the output should be like [4+cls, box_cnt]
    # cls means count of classes
    # box_cnt means count of bboxes
    max_scores = output[4:, :].max(axis=0)
    output = output[:, max_scores > conf_threshold].transpose(1, 0)
    boxes = output[:, :4]
    scores = output[:, 4:]
    filtered_max_scores = scores.max(axis=1)

    if not boxes.size:
        return []

    boxes = _yolo_xywh2xyxy(boxes)
    idx = _yolo_nms(boxes, filtered_max_scores, iou_threshold=iou_threshold)
    boxes, scores = boxes[idx], scores[idx]

    detections = []
    for box, score in zip(boxes, scores):
        x0, y0 = _xy_postprocess(box[0], box[1], old_size, new_size)
        x1, y1 = _xy_postprocess(box[2], box[3], old_size, new_size)
        max_score_id = score.argmax()
        detections.append(((x0, y0, x1, y1), labels[max_score_id], float(score[max_score_id])))

    return detections


def _yolo_postprocess(output, conf_threshold: float, iou_threshold: float,
                      old_size: Tuple[float, float], new_size: Tuple[float, float], labels: List[str]) \
        -> List[Tuple[Tuple[int, int, int, int], str, float]]:
    """
    Post-process the raw output from the object detection model.

    This function applies confidence thresholding, non-maximum suppression, and
    converts the coordinates back to the original image size.

    :param output: Raw output from the object detection model.
    :type output: np.ndarray
    :param conf_threshold: Confidence threshold for filtering detections.
    :type conf_threshold: float
    :param iou_threshold: IoU threshold for non-maximum suppression.
    :type iou_threshold: float
    :param old_size: Original image dimensions (width, height).
    :type old_size: Tuple[float, float]
    :param new_size: Preprocessed image dimensions (width, height).
    :type new_size: Tuple[float, float]
    :param labels: List of class labels.
    :type labels: List[str]

    :return: List of detections, each in the format ((x0, y0, x1, y1), label, confidence).
    :rtype: List[tuple(tuple(int, int, int, int), str, float)]

    :Example:

    >>> output = np.array([[10, 10, 20, 20, 0.9, 0.1]])
    >>> _yolo_postprocess(output, 0.5, 0.5, (100, 100), (128, 128), ['cat', 'dog'])
    [((7, 7, 15, 15), 'cat', 0.9)]
    """
    if output.shape[-1] == 6:  # for end-to-end models like yolov10
        return _end2end_postprocess(
            output=output,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            old_size=old_size,
            new_size=new_size,
            labels=labels,
        )
    else:  # for nms-based models like yolov8
        return _nms_postprocess(
            output=output,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            old_size=old_size,
            new_size=new_size,
            labels=labels,
        )


def _rtdetr_postprocess(output, conf_threshold: float, iou_threshold: float,
                        old_size: Tuple[int, int], new_size: Tuple[int, int], labels: List[str]) \
        -> List[Tuple[Tuple[int, int, int, int], str, float]]:
    """
    Post-process the output from an RT-DETR (Real-Time DEtection TRansformer) model.

    This function handles the specific output format of RT-DETR models and applies
    the necessary post-processing steps.

    :param output: Raw output from the RT-DETR model.
    :type output: np.ndarray
    :param conf_threshold: Confidence threshold for filtering detections.
    :type conf_threshold: float
    :param iou_threshold: IoU threshold for non-maximum suppression.
    :type iou_threshold: float
    :param old_size: Original image dimensions (width, height).
    :type old_size: Tuple[int, int]
    :param new_size: Preprocessed image dimensions (width, height) (not used in this function).
    :type new_size: Tuple[int, int]
    :param labels: List of class labels.
    :type labels: List[str]

    :return: List of detections, each in the format ((x0, y0, x1, y1), label, confidence).
    :rtype: List[Tuple[Tuple[int, int, int, int], str, float]]

    :raises AssertionError: If the output shape is not as expected.
    """
    assert output.shape[-1] == 4 + len(labels)
    # the size rtdetr using is [0.0, 1.0]
    _ = new_size
    return _nms_postprocess(
        output=output.transpose(1, 0),
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        old_size=old_size,
        new_size=(1.0, 1.0),
        labels=labels,
    )


def _safe_eval_names_str(names_str):
    """
    Safely evaluate the names string from model metadata.

    :param names_str: String representation of names dictionary.
    :type names_str: str
    :return: Dictionary of name mappings.
    :rtype: dict
    :raises RuntimeError: If an invalid key or value type is encountered.

    This function parses the names string from the model metadata, ensuring that
    only string and number literals are evaluated for safety.
    """
    node = ast.parse(names_str, mode='eval')
    result = {}
    # noinspection PyUnresolvedReferences
    for key, value in zip(node.body.keys, node.body.values):
        if isinstance(key, (ast.Str, ast.Num)):
            key = ast.literal_eval(key)
        else:
            raise RuntimeError(f"Invalid key type: {key!r}, this should be a bug, "
                               f"please open an issue to dghs-imgutils.")  # pragma: no cover

        if isinstance(value, (ast.Str, ast.Num)):
            value = ast.literal_eval(value)
        else:
            raise RuntimeError(f"Invalid value type: {value!r}, this should be a bug, "
                               f"please open an issue to dghs-imgutils.")  # pragma: no cover

        result[key] = value

    return result


class YOLOModel:
    """
    A class to manage YOLO models from a Hugging Face repository.

    This class handles the loading, caching, and inference of YOLO models.

    :param repo_id: The Hugging Face repository ID containing the YOLO models.
    :type repo_id: str
    :param hf_token: Optional Hugging Face authentication token.
    :type hf_token: Optional[str]

    :Example:

    >>> model = YOLOModel("username/repo_name")
    >>> image = Image.open("path/to/image.jpg")
    >>> detections = model.predict(image, "model_name")
    """

    def __init__(self, repo_id: str, hf_token: Optional[str] = None):
        """
        Initialize the YOLOModel.

        :param repo_id: The Hugging Face repository ID containing the YOLO models.
        :type repo_id: str
        :param hf_token: Optional Hugging Face authentication token.
        :type hf_token: Optional[str]
        """
        self.repo_id = repo_id
        self._model_names = None
        self._models = {}
        self._model_types = {}
        self._hf_token = hf_token
        self._global_lock = Lock()
        self._model_lock = Lock()

    def _get_hf_token(self) -> Optional[str]:
        """
        Get the Hugging Face token, either from the instance or environment variable.

        :return: Hugging Face token.
        :rtype: Optional[str]
        """
        return self._hf_token or os.environ.get('HF_TOKEN')

    @property
    def model_names(self) -> List[str]:
        """
        Get the list of available model names in the repository.

        :return: List of model names.
        :rtype: List[str]
        """
        with self._global_lock:
            if self._model_names is None:
                hf_fs = HfFileSystem(token=self._get_hf_token())
                self._model_names = [
                    hf_normpath(os.path.dirname(os.path.relpath(item, self.repo_id)))
                    for item in hf_fs.glob(hf_fs_path(
                        repo_id=self.repo_id,
                        repo_type='model',
                        filename='*/model.onnx',
                    ))
                ]

        return self._model_names

    def _check_model_name(self, model_name: str):
        """
        Check if the given model name is valid for this repository.

        :param model_name: Name of the model to check.
        :type model_name: str
        :raises ValueError: If the model name is not found in the repository.
        """
        if model_name not in self.model_names:
            raise ValueError(f'Unknown model {model_name!r} in model repository {self.repo_id!r}, '
                             f'models {self.model_names!r} are available.')

    def _open_model(self, model_name: str):
        """
        Open and cache a YOLO model.

        :param model_name: Name of the model to open.
        :type model_name: str
        :return: Tuple containing the ONNX model, maximum inference size, and labels.
        :rtype: tuple
        """
        with self._model_lock:
            if model_name not in self._models:
                self._check_model_name(model_name)
                model = open_onnx_model(hf_hub_download(
                    self.repo_id,
                    f'{model_name}/model.onnx',
                    token=self._get_hf_token(),
                ))
                model_metadata = model.get_modelmeta()
                if 'imgsz' in model_metadata.custom_metadata_map:
                    max_infer_size = tuple(json.loads(model_metadata.custom_metadata_map['imgsz']))
                    assert len(max_infer_size) == 2, f'imgsz should have 2 dims, but {max_infer_size!r} found.'
                else:
                    max_infer_size = 640
                names_map = _safe_eval_names_str(model_metadata.custom_metadata_map['names'])
                labels = [names_map[i] for i in range(len(names_map))]
                self._models[model_name] = (model, max_infer_size, labels)

        return self._models[model_name]

    def _get_model_type(self, model_name: str):
        with self._model_lock:
            if model_name not in self._model_types:
                hf_fs = get_hf_fs(hf_token=self._get_hf_token())
                fs_path = hf_fs_path(
                    repo_id=self.repo_id,
                    repo_type='model',
                    filename=f'{model_name}/model_type.json',
                    revision='main',
                )
                if hf_fs.exists(fs_path):
                    model_type = json.loads(hf_fs.read_text(fs_path))['model_type']
                else:
                    model_type = 'yolo'
                self._model_types[model_name] = model_type

        return self._model_types[model_name]

    def predict(self, image: ImageTyping, model_name: str,
                conf_threshold: float = 0.25, iou_threshold: float = 0.7,
                allow_dynamic: bool = False) \
            -> List[Tuple[Tuple[int, int, int, int], str, float]]:
        """
        Perform object detection on an image using the specified YOLO model.

        :param image: Input image for object detection.
        :type image: ImageTyping
        :param model_name: Name of the YOLO model to use.
        :type model_name: str
        :param conf_threshold: Confidence threshold for filtering detections. Default is 0.25.
        :type conf_threshold: float
        :param iou_threshold: IoU threshold for non-maximum suppression. Default is 0.7.
        :type iou_threshold: float
        :param allow_dynamic: If True, allows dynamic resizing of the image while maintaining aspect ratio.
                              Default is False.
        :type allow_dynamic: bool

        :return: List of detections, each in the format ((x0, y0, x1, y1), label, confidence).
        :rtype: List[Tuple[Tuple[int, int, int, int], str, float]]

        :Example:

        >>> model = YOLOModel("username/repo_name")
        >>> image = Image.open("path/to/image.jpg")
        >>> detections = model.predict(image, "model_name")
        >>> print(detections[0])  # First detection
        ((100, 200, 300, 400), 'person', 0.95)
        """
        model, max_infer_size, labels = self._open_model(model_name)
        image = load_image(image, mode='RGB')
        new_image, old_size, new_size = _image_preprocess(image, max_infer_size, allow_dynamic=allow_dynamic)
        data = rgb_encode(new_image)[None, ...]
        output, = model.run(['output0'], {'images': data})
        model_type = self._get_model_type(model_name=model_name)
        if model_type == 'yolo':
            return _yolo_postprocess(
                output=output[0],
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                old_size=old_size,
                new_size=new_size,
                labels=labels
            )
        elif model_type == 'rtdetr':
            return _rtdetr_postprocess(
                output=output[0],
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                old_size=old_size,
                new_size=new_size,
                labels=labels
            )
        else:
            raise ValueError(f'Unknown object detection model type - {model_type!r}.')  # pragma: no cover

    def clear(self):
        """
        Clear cached model and metadata.

        This method removes all cached models and their associated metadata from memory.
        It's useful for freeing up memory or ensuring that the latest versions of models are loaded.
        """
        self._models.clear()

    def make_ui(self, default_model_name: Optional[str] = None,
                default_conf_threshold: float = 0.25, default_iou_threshold: float = 0.7):
        """
        Create a Gradio-based user interface for object detection.

        This method sets up an interactive UI that allows users to upload images,
        select models, and adjust detection parameters. It uses the Gradio library
        to create the interface.

        :param default_model_name: The name of the default model to use.
                                   If None, the most recently updated model is selected.
        :type default_model_name: Optional[str]
        :param default_conf_threshold: Default confidence threshold for the UI. Default is 0.25.
        :type default_conf_threshold: float
        :param default_iou_threshold: Default IoU threshold for the UI. Default is 0.7.
        :type default_iou_threshold: float

        :raises ImportError: If Gradio is not installed in the environment.

        :Example:

        >>> model = YOLOModel("username/repo_name")
        >>> model.make_ui(default_model_name="yolov5s")
        """
        _check_gradio_env()
        model_list = self.model_names
        if not default_model_name:
            hf_client = get_hf_client(hf_token=self._get_hf_token())
            selected_model_name, selected_time = None, None
            for fileitem in hf_client.get_paths_info(
                    repo_id=self.repo_id,
                    repo_type='model',
                    paths=[f'{model_name}/model.onnx' for model_name in model_list],
                    expand=True,
            ):
                if not selected_time or fileitem.last_commit.date > selected_time:
                    selected_model_name = os.path.dirname(fileitem.path)
                    selected_time = fileitem.last_commit.date
            default_model_name = selected_model_name

        def _gr_detect(image: ImageTyping, model_name: str,
                       iou_threshold: float = 0.7, score_threshold: float = 0.25,
                       allow_dynamic: bool = False) \
                -> gr.AnnotatedImage:
            _, _, labels = self._open_model(model_name=model_name)
            _colors = list(map(str, rnd_colors(len(labels))))
            _color_map = dict(zip(labels, _colors))
            return gr.AnnotatedImage(
                value=(image, [
                    (_bbox_fix(bbox), label)
                    for bbox, label, _ in self.predict(
                        image=image,
                        model_name=model_name,
                        iou_threshold=iou_threshold,
                        conf_threshold=score_threshold,
                        allow_dynamic=allow_dynamic,
                    )
                ]),
                color_map=_color_map,
                label='Labeled',
            )

        with gr.Row():
            with gr.Column():
                gr_input_image = gr.Image(type='pil', label='Original Image')
                with gr.Row():
                    gr_model = gr.Dropdown(model_list, value=default_model_name, label='Model')
                    gr_allow_dynamic = gr.Checkbox(value=False, label='Allow Dynamic Size')
                with gr.Row():
                    gr_iou_threshold = gr.Slider(0.0, 1.0, default_iou_threshold, label='IOU Threshold')
                    gr_score_threshold = gr.Slider(0.0, 1.0, default_conf_threshold, label='Score Threshold')

                gr_submit = gr.Button(value='Submit', variant='primary')

            with gr.Column():
                gr_output_image = gr.AnnotatedImage(label="Labeled")

            gr_submit.click(
                _gr_detect,
                inputs=[
                    gr_input_image,
                    gr_model,
                    gr_iou_threshold,
                    gr_score_threshold,
                    gr_allow_dynamic,
                ],
                outputs=[gr_output_image],
            )

    def launch_demo(self, default_model_name: Optional[str] = None,
                    default_conf_threshold: float = 0.25, default_iou_threshold: float = 0.7,
                    server_name: Optional[str] = None, server_port: Optional[int] = None, **kwargs):
        """
        Launch a Gradio demo for object detection.

        This method creates and launches a Gradio demo that allows users to interactively
        perform object detection on uploaded images using the YOLO model.

        :param default_model_name: The name of the default model to use.
                                   If None, the most recently updated model is selected.
        :type default_model_name: Optional[str]
        :param default_conf_threshold: Default confidence threshold for the demo. Default is 0.25.
        :type default_conf_threshold: float
        :param default_iou_threshold: Default IoU threshold for the demo. Default is 0.7.
        :type default_iou_threshold: float
        :param server_name: The name of the server to run the demo on. Default is None.
        :type server_name: Optional[str]
        :param server_port: The port to run the demo on. Default is None.
        :type server_port: Optional[int]
        :param kwargs: Additional keyword arguments to pass to gr.Blocks.launch().

        :raises EnvironmentError: If Gradio is not installed in the environment.

        Example:
            >>> model = YOLOModel("username/repo_name")
            >>> model.launch_demo(default_model_name="yolov5s", server_name="0.0.0.0", server_port=7860)
        """
        _check_gradio_env()
        with gr.Blocks() as demo:
            with gr.Row():
                with gr.Column():
                    repo_url = hf_hub_repo_url(repo_id=self.repo_id, repo_type='model')
                    gr.HTML(f'<h2 style="text-align: center;">YOLO Demo For {self.repo_id}</h2>')
                    gr.Markdown(f'This is the quick demo for YOLO model [{self.repo_id}]({repo_url}). '
                                f'Powered by `dghs-imgutils`\'s quick demo module.')

            with gr.Row():
                self.make_ui(
                    default_model_name=default_model_name,
                    default_conf_threshold=default_conf_threshold,
                    default_iou_threshold=default_iou_threshold,
                )

        demo.launch(
            server_name=server_name,
            server_port=server_port,
            **kwargs,
        )


@ts_lru_cache()
def _open_models_for_repo_id(repo_id: str, hf_token: Optional[str] = None) -> YOLOModel:
    """
    Load and cache a YOLO model from a Hugging Face repository.

    This function uses the `lru_cache` decorator to cache the loaded models,
    improving performance for repeated calls with the same repository ID.

    :param repo_id: The Hugging Face repository ID for the YOLO model.
    :type repo_id: str
    :param hf_token: Optional Hugging Face authentication token.
    :type hf_token: Optional[str]

    :return: The loaded YOLO model.
    :rtype: YOLOModel

    :raises Exception: If there's an error loading the model from the repository.

    Usage:
        >>> model = _open_models_for_repo_id("yolov5/yolov5s")
        >>> # Subsequent calls with the same repo_id will return the cached model
        >>> same_model = _open_models_for_repo_id("yolov5/yolov5s")
    """
    return YOLOModel(repo_id, hf_token=hf_token)


def yolo_predict(image: ImageTyping, repo_id: str, model_name: str,
                 conf_threshold: float = 0.25, iou_threshold: float = 0.7,
                 hf_token: Optional[str] = None, **kwargs) \
        -> List[Tuple[Tuple[int, int, int, int], str, float]]:
    """
    Perform object detection on an image using a YOLO model from a Hugging Face repository.

    This function is a high-level wrapper around the YOLOModel class, providing a simple
    interface for object detection without needing to explicitly manage model instances.

    :param image: Input image for object detection.
    :type image: ImageTyping
    :param repo_id: The Hugging Face repository ID containing the YOLO models.
    :type repo_id: str
    :param model_name: Name of the YOLO model to use.
    :type model_name: str
    :param conf_threshold: Confidence threshold for filtering detections. Default is 0.25.
    :type conf_threshold: float
    :param iou_threshold: IoU threshold for non-maximum suppression. Default is 0.7.
    :type iou_threshold: float
    :param hf_token: Optional Hugging Face authentication token.
    :type hf_token: Optional[str]

    :return: List of detections, each in the format ((x0, y0, x1, y1), label, confidence).
    :rtype: List[Tuple[Tuple[int, int, int, int], str, float]]

    :Example:

    >>> from PIL import Image
    >>> image = Image.open("path/to/image.jpg")
    >>> detections = yolo_predict(image, "username/repo_name", "model_name")
    >>> print(detections[0])  # First detection
    ((100, 200, 300, 400), 'person', 0.95)
    """
    return _open_models_for_repo_id(repo_id, hf_token=hf_token).predict(
        image=image,
        model_name=model_name,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        **kwargs,
    )
