"""
YOLO Segmentation Module for Image Processing

This module provides functionality for YOLO-based segmentation models, allowing users to 
perform instance segmentation on images. It includes classes and functions for loading models 
from Hugging Face repositories, making predictions, and creating interactive demos.

The module supports both online and offline operations, with thread-safe model loading and 
execution. It handles various image formats and provides utilities for pre-processing and 
post-processing segmentation results.
"""

import json
import os
import threading
from collections import defaultdict
from contextlib import contextmanager
from threading import Lock
from typing import List, Optional, Tuple, Literal

import cv2
import numpy as np
import requests
from hbutils.color import rnd_colors
from hfutils.operate import get_hf_fs, get_hf_client
from hfutils.repository import hf_hub_repo_url
from hfutils.utils import hf_fs_path, hf_normpath
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import OfflineModeIsEnabled, EntryNotFoundError

from .yolo import _OFFLINE, _safe_eval_names_str, _image_preprocess, _yolo_xywh2xyxy, _yolo_nms, _xy_postprocess
from ..data import load_image, rgb_encode, ImageTyping
from ..utils import open_onnx_model, ts_lru_cache

try:
    import gradio as gr
except (ImportError, ModuleNotFoundError):
    gr = None

__all__ = [
    'YOLOSegmentationModel',
    'yolo_seg_predict',
]


def _check_gradio_env():
    """
    Check if the Gradio library is installed and available.

    :raises EnvironmentError: If Gradio is not installed.
    """
    if gr is None:
        raise EnvironmentError(f'Gradio required for launching webui-based demo.\n'
                               f'Please install it with `pip install dghs-imgutils[demo]`.')


_MODEL_LOAD_LOCKS = defaultdict(Lock)
_G_ML_LOCK = Lock()


@contextmanager
def _model_load_lock():
    """
    Context manager for thread-safe model loading operations.

    This context manager ensures that model loading operations are thread-safe by using
    process-specific locks. It prevents concurrent model loading operations which could
    lead to race conditions.

    :yields: None
    """
    with _G_ML_LOCK:
        lock = _MODEL_LOAD_LOCKS[os.getpid()]
    with lock:
        yield


def crop_mask(masks, boxes):
    """
    Crop masks to bounding box regions.

    :param masks: Masks with shape (H, W).
    :type masks: numpy.ndarray
    :param boxes: Bounding box coordinates with shape (4, ) in relative point form.
    :type boxes: numpy.ndarray

    :return: Cropped masks.
    :rtype: numpy.ndarray
    """
    h, w = masks.shape
    x1, y1, x2, y2 = np.split(boxes[:, None], 4)  # x1 shape(1,1)
    r = np.arange(w)[None, :]  # rows shape(1,w)
    c = np.arange(h)[:, None]  # cols shape(h,1)

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


def scale_masks(masks, shape, padding: Literal['none', 'center', 'left'] = 'none'):
    """
    Rescale segment masks to target shape.

    :param masks: Masks with shape (H, W).
    :type masks: numpy.ndarray
    :param shape: Target height and width as (height, width).
    :type shape: tuple
    :param padding: Type of padding applied to masks. Options are 'none', 'center', or 'left'.
    :type padding: Literal['none', 'center', 'left']

    :return: Rescaled masks.
    :rtype: numpy.ndarray
    """
    mh, mw = masks.shape
    if padding != 'none':
        gain = min(mh / shape[0], mw / shape[1])  # gain = old / new
        pad = [mw - shape[1] * gain, mh - shape[0] * gain]  # wh padding
        if padding == 'center':
            pad[0], pad[1] = pad[0] / 2, pad[1] / 2
        top, left = (int(round(pad[1] - 0.1)), int(round(pad[0] - 0.1))) if padding else (0, 0)  # y, x
        bottom, right = mh - int(round(pad[1] + 0.1)), mw - int(round(pad[0] + 0.1))
        masks = masks[top:bottom, left:right]

    resized_masks = cv2.resize(masks, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return resized_masks


def _nms_postprocess(output: np.ndarray, protos: np.ndarray, conf_threshold: float, iou_threshold: float,
                     old_size: Tuple[float, float], new_size: Tuple[float, float], labels: List[str]) \
        -> List[Tuple[Tuple[int, int, int, int], str, float, np.ndarray]]:
    """
    Perform non-maximum suppression (NMS) post-processing on YOLO segmentation output.

    :param output: Raw output from YOLO model with shape [4+cls+pe, box_cnt].
    :type output: numpy.ndarray
    :param protos: Prototype masks from YOLO model.
    :type protos: numpy.ndarray
    :param conf_threshold: Confidence threshold for filtering detections.
    :type conf_threshold: float
    :param iou_threshold: IoU threshold for NMS.
    :type iou_threshold: float
    :param old_size: Original image size (width, height).
    :type old_size: Tuple[float, float]
    :param new_size: New image size after preprocessing (width, height).
    :type new_size: Tuple[float, float]
    :param labels: List of class labels.
    :type labels: List[str]

    :return: List of detections, each containing bounding box, class label, confidence score, and mask.
    :rtype: List[Tuple[Tuple[int, int, int, int], str, float, numpy.ndarray]]
    """
    pe, pheight, pwidth = protos.shape
    assert output.shape[0] == 4 + len(labels) + pe
    # the output should be like [4+cls+pe, box_cnt]
    # cls means count of classes
    # box_cnt means count of bboxes
    max_scores = output[4:4 + len(labels), :].max(axis=0)
    output = output[:, max_scores > conf_threshold].transpose(1, 0)
    boxes = output[:, :4]
    scores = output[:, 4:4 + len(labels)]
    mask_embeddings = output[:, 4 + len(labels):]
    assert mask_embeddings.shape[-1] == pe
    filtered_max_scores = scores.max(axis=1)

    if not boxes.size:
        return []

    boxes = _yolo_xywh2xyxy(boxes)
    idx = _yolo_nms(boxes, filtered_max_scores, iou_threshold=iou_threshold)
    boxes, scores, mask_embeddings = boxes[idx], scores[idx], mask_embeddings[idx]

    detections = []
    for box, score, mask_embedding in zip(boxes, scores, mask_embeddings):
        x0, y0 = _xy_postprocess(box[0], box[1], old_size, new_size)
        x1, y1 = _xy_postprocess(box[2], box[3], old_size, new_size)
        max_score_id = score.argmax()
        mask = (mask_embedding @ protos.reshape(pe, pheight * pwidth)).reshape(pheight, pwidth)
        mask = scale_masks(mask, shape=(old_size[1], old_size[0]))  # CHW
        mask = crop_mask(mask, np.array([x0, y0, x1, y1]))  # CHW
        mask = (mask > 0.0).astype(np.float32)
        detections.append(((x0, y0, x1, y1), labels[max_score_id], float(score[max_score_id]), mask))

    return detections


def _yolo_seg_postprocess(output: np.ndarray, protos: np.ndarray, conf_threshold: float, iou_threshold: float,
                          old_size: Tuple[float, float], new_size: Tuple[float, float], labels: List[str]) \
        -> List[Tuple[Tuple[int, int, int, int], str, float, np.ndarray]]:
    """
    Post-process YOLO segmentation model output.

    :param output: Raw output from YOLO model.
    :type output: numpy.ndarray
    :param protos: Prototype masks from YOLO model.
    :type protos: numpy.ndarray
    :param conf_threshold: Confidence threshold for filtering detections.
    :type conf_threshold: float
    :param iou_threshold: IoU threshold for NMS.
    :type iou_threshold: float
    :param old_size: Original image size (width, height).
    :type old_size: Tuple[float, float]
    :param new_size: New image size after preprocessing (width, height).
    :type new_size: Tuple[float, float]
    :param labels: List of class labels.
    :type labels: List[str]

    :return: List of detections, each containing bounding box, class label, confidence score, and mask.
    :rtype: List[Tuple[Tuple[int, int, int, int], str, float, numpy.ndarray]]
    """
    return _nms_postprocess(
        output=output,
        protos=protos,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        old_size=old_size,
        new_size=new_size,
        labels=labels,
    )


class YOLOSegmentationModel:
    """
    YOLO-based segmentation model loaded from Hugging Face repositories.

    This class provides functionality for loading YOLO segmentation models from Hugging Face,
    making predictions, and creating interactive UIs for model demonstration.

    :param repo_id: Hugging Face repository ID containing the YOLO segmentation models.
    :type repo_id: str
    :param hf_token: Hugging Face API token for accessing private repositories.
                    If None, will try to use the HF_TOKEN environment variable.
    :type hf_token: Optional[str]
    """

    def __init__(self, repo_id: str, hf_token: Optional[str] = None):
        """
        Initialize a YOLO segmentation model.

        :param repo_id: Hugging Face repository ID containing the YOLO segmentation models.
        :type repo_id: str
        :param hf_token: Hugging Face API token for accessing private repositories.
                        If None, will try to use the HF_TOKEN environment variable.
        :type hf_token: Optional[str]
        """
        self.repo_id = repo_id
        self._model_names = None
        self._models = {}
        self._model_types = {}
        self._hf_token = hf_token
        self._global_lock = Lock()
        self._model_meta_lock = Lock()

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

        This property performs a glob search in the Hugging Face repository to find all ONNX models.
        The search is thread-safe and implements caching to avoid repeated filesystem operations.
        Results are normalized to provide consistent path formats.

        :return: List of available model names in the repository. Returns _OFFLINE list if offline mode is enabled
                or connection errors occur.
        :rtype: List[str]
        """
        with self._global_lock:
            if self._model_names is None:
                try:
                    hf_fs = get_hf_fs(hf_token=self._get_hf_token())
                    self._model_names = [
                        hf_normpath(os.path.dirname(os.path.relpath(item, self.repo_id)))
                        for item in hf_fs.glob(hf_fs_path(
                            repo_id=self.repo_id,
                            repo_type='model',
                            filename='*/model.onnx',
                        ))
                    ]
                except (
                        requests.exceptions.ConnectionError,
                        requests.exceptions.Timeout,
                        OfflineModeIsEnabled,
                ):
                    self._model_names = _OFFLINE

        return self._model_names

    def _check_model_name(self, model_name: str):
        """
        Check if the given model name is valid for this repository.

        This method validates model names against the available models in the repository.
        Validation is skipped in offline mode to allow for local operations.

        :param model_name: Name of the model to check against the repository's available models.
        :type model_name: str
        :raises ValueError: If the model name is not found in the repository and not in offline mode.
                           The error message includes available model names for reference.
        :note: This method is a helper function primarily used internally for model validation.
        """
        model_list = self.model_names
        if model_list is _OFFLINE:
            return  # do not check when in offline mode
        if model_name not in model_list:
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
        cache_key = os.getpid(), threading.get_ident(), model_name
        with _model_load_lock():
            if cache_key not in self._models:
                self._check_model_name(model_name)
                model = open_onnx_model(hf_hub_download(
                    repo_id=self.repo_id,
                    repo_type='model',
                    filename=f'{model_name}/model.onnx',
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
                self._models[cache_key] = (model, max_infer_size, labels, Lock())

        return self._models[cache_key]

    def _get_model_type(self, model_name: str):
        """
        Get the type of the specified model.

        :param model_name: Name of the model to get the type for.
        :type model_name: str
        :return: Model type string.
        :rtype: str
        """
        with self._model_meta_lock:
            if model_name not in self._model_types:
                try:
                    model_type_file = hf_hub_download(
                        repo_id=self.repo_id,
                        repo_type='model',
                        filename=f'{model_name}/model_type.json',
                        revision='main',
                        token=self._get_hf_token()
                    )
                except (EntryNotFoundError,):
                    model_type = 'yolo'
                else:
                    with open(model_type_file, 'r') as f:
                        model_type = json.load(f)['model_type']

                self._model_types[model_name] = model_type

        return self._model_types[model_name]

    def predict(self, image: ImageTyping, model_name: str,
                conf_threshold: float = 0.25, iou_threshold: float = 0.7,
                allow_dynamic: bool = False) \
            -> List[Tuple[Tuple[int, int, int, int], str, float, np.ndarray]]:
        """
        Perform segmentation prediction on an image.

        :param image: Input image to perform segmentation on.
        :type image: ImageTyping
        :param model_name: Name of the model to use for prediction.
        :type model_name: str
        :param conf_threshold: Confidence threshold for filtering detections (0.0-1.0).
        :type conf_threshold: float
        :param iou_threshold: IoU threshold for non-maximum suppression (0.0-1.0).
        :type iou_threshold: float
        :param allow_dynamic: Whether to allow dynamic resizing of the input image.
        :type allow_dynamic: bool

        :return: List of detections, each containing bounding box, class label, confidence score, and mask.
        :rtype: List[Tuple[Tuple[int, int, int, int], str, float, numpy.ndarray]]

        :raises ValueError: If the model type is unknown.

        :Example:

        >>> model = YOLOSegmentationModel("username/repo_name")
        >>> results = model.predict(
        ...     image="path/to/image.jpg",
        ...     model_name="yolov8s-seg",
        ...     conf_threshold=0.3
        ... )
        >>> for bbox, label, confidence, mask in results:
        ...     print(f"Found {label} with confidence {confidence:.2f}")
        """
        model, max_infer_size, labels, exec_lock = self._open_model(model_name)
        image = load_image(image, mode='RGB')
        new_image, old_size, new_size = _image_preprocess(image, max_infer_size, allow_dynamic=allow_dynamic)
        data = rgb_encode(new_image)[None, ...]
        with exec_lock:  # make sure for each session, its execution should be linear
            output, protos = model.run(['output0', 'output1'], {'images': data})
        model_type = self._get_model_type(model_name=model_name)
        if model_type == 'yolo':
            return _yolo_seg_postprocess(
                output=output[0],
                protos=protos[0],
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
        Clear all cached models and metadata.

        This method resets the model cache, forcing new model loads on subsequent operations.
        It's useful for freeing memory or when switching between different models.
        """
        self._model_names = None
        self._models.clear()
        self._model_types.clear()

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
        :raises EnvironmentError: If in OFFLINE mode and no default_model_name is provided.

        :Example:

        >>> model = YOLOSegmentationModel("username/repo_name")
        >>> model.make_ui(default_model_name="yolov8s-seg")
        """
        _check_gradio_env()
        model_list = self.model_names
        if model_list is _OFFLINE and not default_model_name:
            raise EnvironmentError('You are in OFFLINE mode, '
                                   'you must assign a default model name to make this ui usable.')

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
            _, _, labels, _ = self._open_model(model_name=model_name)
            _colors = list(map(str, rnd_colors(len(labels))))
            _color_map = dict(zip(labels, _colors))
            return gr.AnnotatedImage(
                value=(image, [
                    (mask, label)
                    for bbox, label, _, mask in self.predict(
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
                    if model_list is not _OFFLINE:
                        gr_model = gr.Dropdown(model_list, value=default_model_name, label='Model')
                    else:
                        gr_model = gr.Dropdown([default_model_name], value=default_model_name, label='Model',
                                               interactive=False)
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

        :raises EnvironmentError: If Gradio is not installed in the environment,
                                  or if in OFFLINE mode and no default_model_name is provided.

        :Example:

            >>> model = YOLOSegmentationModel("username/repo_name")
            >>> model.launch_demo(default_model_name="yolov8s-seg", server_name="0.0.0.0", server_port=7860)
        """
        _check_gradio_env()
        with gr.Blocks() as demo:
            with gr.Row():
                with gr.Column():
                    repo_url = hf_hub_repo_url(repo_id=self.repo_id, repo_type='model')
                    gr.HTML(f'<h2 style="text-align: center;">YOLO-Seg Demo For {self.repo_id}</h2>')
                    gr.Markdown(f'This is the quick demo for YOLO-Seg model [{self.repo_id}]({repo_url}). '
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
def _open_models_for_repo_id(repo_id: str, hf_token: Optional[str] = None) -> YOLOSegmentationModel:
    """
    Open and cache a YOLOSegmentationModel for a specific repository.

    This function uses thread-safe LRU caching to avoid repeatedly creating model instances
    for the same repository ID.

    :param repo_id: Hugging Face repository ID.
    :type repo_id: str
    :param hf_token: Hugging Face API token.
    :type hf_token: Optional[str]

    :return: Cached YOLOSegmentationModel instance.
    :rtype: YOLOSegmentationModel
    """
    return YOLOSegmentationModel(repo_id, hf_token=hf_token)


def yolo_seg_predict(image: ImageTyping, repo_id: str, model_name: str,
                     conf_threshold: float = 0.25, iou_threshold: float = 0.7,
                     hf_token: Optional[str] = None, **kwargs) \
        -> List[Tuple[Tuple[int, int, int, int], str, float, np.ndarray]]:
    """
    Perform YOLO segmentation prediction using a model from Hugging Face.

    This is a convenience function that creates a YOLOSegmentationModel instance
    and performs prediction in one step.

    :param image: Input image to perform segmentation on.
    :type image: ImageTyping
    :param repo_id: Hugging Face repository ID containing the model.
    :type repo_id: str
    :param model_name: Name of the specific model to use.
    :type model_name: str
    :param conf_threshold: Confidence threshold for filtering detections (0.0-1.0).
    :type conf_threshold: float
    :param iou_threshold: IoU threshold for non-maximum suppression (0.0-1.0).
    :type iou_threshold: float
    :param hf_token: Hugging Face API token for accessing private repositories.
    :type hf_token: Optional[str]
    :param kwargs: Additional keyword arguments to pass to the predict method.

    :return: List of detections, each containing bounding box, class label, confidence score, and mask.
    :rtype: List[Tuple[Tuple[int, int, int, int], str, float, numpy.ndarray]]

    :Example:

    >>> results = yolo_seg_predict(
    ...     image="path/to/image.jpg",
    ...     repo_id="username/repo_name",
    ...     model_name="yolov8s-seg",
    ...     conf_threshold=0.3
    ... )
    >>> for bbox, label, confidence, mask in results:
    ...     print(f"Found {label} with confidence {confidence:.2f}")
    """
    return _open_models_for_repo_id(repo_id, hf_token=hf_token).predict(
        image=image,
        model_name=model_name,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        **kwargs,
    )
