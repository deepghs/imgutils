"""
Overview:
    This module provides functionality for object detection using the Booru YOLO model.

    The Booru YOLO model is sourced from `aperveyev/booru_yolo <https://github.com/aperveyev/booru_yolo>`_ and
    model files are hosted on `deepghs/booru_yolo <https://huggingface.co/deepghs/booru_yolo>`_.

    .. collapse:: Overview of Booru YOLO Detect (NSFW Warning!!!)

        .. image:: booru_yolo_detect_demo.plot.py.svg
            :align: center

    This is an overall benchmark of all the booru yolo models:

    .. image:: booru_yolo_detect_benchmark.plot.py.svg
        :align: center

    Here is the explanations of each label:

    .. list-table:: Model Labels and Descriptions
       :widths: 10 20 70
       :header-rows: 1

       * - No.
         - Label
         - Description
       * - 0
         - head
         - Anime pretty girl and not only
       * - 1
         - bust
         - Torso part from collarbone center to pair of covered breasts
       * - 2
         - boob
         - Bust with no bra, nipples mostly visible, generally NSFW
       * - 3
         - shld
         - Shoulder and maybe one breast viewed mostly in profile, exactly rear view excluded
       * - 4
         - sideb
         - Uncovered version of shld, with nipples or other NSFW visual marks
       * - 5
         - belly
         - From belly button to hips half (stocking line), knees below belly, mostly covered
       * - 6
         - nopan
         - No panty-like clothes on bikini area (regardless of censoring), evidently NSFW belly
       * - 7
         - butt
         - Buttock area visible at least partially from behind, more or less covered, standing or sitting
       * - 8
         - ass
         - Uncovered NSFW version of butt
       * - 9
         - split
         - Sitting with legs open wide (90+ degrees), typically with at least one knee above belly
       * - 10
         - sprd
         - Strongly NSFW version of split
       * - 11
         - vsplt
         - Stand split or visually similar pose
       * - 12
         - vsprd
         - Strongly NSFW version of vsplit
       * - 13
         - hip
         - Full or almost full hip(-s) side view with knee(-s) above belly, usually when sitting or lying
       * - 14
         - wing
         - Mostly dragon or pony related
       * - 15
         - feral
         - All-four non-human torso
       * - 16
         - hdrago
         - Dragon style head
       * - 17
         - hpony
         - Pony style head
       * - 18
         - hfox
         - Cartoon fox / dog head - Zootopia Nick Wilde
       * - 19
         - hrabb
         - Cartoon rabbit head - Zootopia Judy Hopps or bunnygirl
       * - 20
         - hcat
         - Cartoon cat or anime catgirl head (less sharp muzzle compared to hfox)
       * - 21
         - hbear
         - Cartoon bear head
       * - 22
         - jacko
         - Memetic "Jack'O contest pose" with a head toward viewer
       * - 23
         - jackx
         - Jacko viewed from behind, sometimes strongly NSFW
       * - 24
         - hhorse
         - Horse head first implemented in aa09
       * - 25
         - hbird
         - Bird head first implemented in aa09

    Here are the list of available models.

    .. list-table:: Available Models (PP Series)
       :widths: 20 20 60
       :header-rows: 1

       * - Model Name
         - Release Time
         - Description
       * - yolov8s_pp09
         - 2023.11
         - PP model focusing on NSFW content with 9 specialized classes, providing advanced detection capabilities
           for specific NSFW scenarios.
       * - yolov8s_pp12
         - 2024.2
         - Final patch for PP models, continuing the focus on specialized NSFW content detection with enhanced
           capabilities and improvements from previous versions.
       * - yolov8m_pp13
         - 2024.2
         - PP model final patch, focusing on specific NSFW content with 9 specialized classes, offering advanced
           NSFW detection capabilities.

    .. list-table:: Available Models (AS Series)
       :widths: 20 20 60
       :header-rows: 1

       * - Model Name
         - Release Time
         - Description
       * - yolov8n_as01
         - 2023.12
         - Spinoff model with 26 classes trained for 80 epochs on an SFW subset. This model was eventually
           abandoned due to ineffective results.
       * - yolov8m_as02
         - 2023.12
         - Spinoff model started using an SFW training subset. It includes 26 classes and was trained for
           30 epochs, offering a more public-friendly model with reduced NSFW content.
       * - yolov8m_as03
         - 2024.1
         - Advanced spinoff model from as02 with 26 classes, trained for 60 epochs. This SFW subset model
           aims to reduce bottlenecks and improve reproducibility of results.

    .. list-table:: Available Models (AA Series)
       :widths: 20 20 60
       :header-rows: 1

       * - Model Name
         - Release Time
         - Description
       * - yolov8s_aa06
         - 2023.8
         - Initial version for current reincarnation with 24 classes trained for 90 epochs. Focuses on general
           torso components and includes some NSFW content.
       * - yolov8s_aa09
         - 2023.10
         - Added HHORSE and HBIRD classes, updating the dataset with more head closeups and adjusting for
           large hats, trained from aa06.
       * - yolov8s_aa10
         - 2023.12
         - Major training dataset update from aa09, focusing on improving detection of HHORSE and HBIRD and
           fixing issues with heads wearing large hats.
       * - yolov8s_aa11
         - 2024.1
         - Latest mainstream general torso components model with 26 classes. It includes an update from aa10
           with outstanding mAP scores, though noted to be not completely fair due to training set biases.

"""
import ast
from functools import lru_cache
from typing import Tuple, List

from huggingface_hub import hf_hub_download

from ._yolo import _data_postprocess, _image_preprocess
from ..data import ImageTyping, load_image, rgb_encode
from ..utils import open_onnx_model


@lru_cache()
def _open_booru_yolo_model(model_name: str):
    """
    Open and cache the Booru YOLO model.

    :param model_name: Name of the model to open.
    :type model_name: str
    :return: Opened ONNX model.
    :rtype: onnxruntime.InferenceSession

    This function downloads the specified model from the Hugging Face Hub and opens it as an ONNX model.
    The result is cached to avoid repeated downloads and openings.
    """
    return open_onnx_model(hf_hub_download(
        repo_id='deepghs/booru_yolo',
        repo_type='model',
        filename=f'{model_name}/model.onnx'
    ))


@lru_cache()
def _get_booru_yolo_labels(model_name: str):
    """
    Retrieve and cache the labels for the Booru YOLO model.

    :param model_name: Name of the model to get labels for.
    :type model_name: str
    :return: List of label names.
    :rtype: List[str]

    This function opens the specified model, extracts the label information from its metadata,
    and returns a list of label names. The result is cached for efficiency.
    """
    model = _open_booru_yolo_model(model_name)
    model_metadata = model.get_modelmeta()
    names_map = _safe_eval_names_str(model_metadata.custom_metadata_map['names'])
    labels = ['<Unknown>'] * (max(names_map.keys()) + 1)
    for id_, name in names_map.items():
        labels[id_] = name
    return labels


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


_DEFAULT_MODEL = 'yolov8s_aa11'


def detect_with_booru_yolo(image: ImageTyping, model_name: str = _DEFAULT_MODEL,
                           max_infer_size: int = 640, conf_threshold: float = 0.25, iou_threshold: float = 0.7) \
        -> List[Tuple[Tuple[int, int, int, int], str, float]]:
    """
    Perform object detection on an image using the Booru YOLO model.

    :param image: Input image to perform detection on.
    :type image: ImageTyping
    :param model_name: Name of the Booru YOLO model to use, defaults to 'yolov8s_aa11'.
    :type model_name: str, optional
    :param max_infer_size: Maximum inference size for image preprocessing, defaults to 640.
    :type max_infer_size: int, optional
    :param conf_threshold: Confidence threshold for detection, defaults to 0.25.
    :type conf_threshold: float, optional
    :param iou_threshold: IOU threshold for non-maximum suppression, defaults to 0.7.
    :type iou_threshold: float, optional
    :return: List of detected objects, each represented as (bounding_box as `(x1, y1, x2, y2)`, label, confidence).
    :rtype: List[Tuple[Tuple[int, int, int, int], str, float]]

    :example:
        >>> from imgutils.detect import detect_with_booru_yolo
        ...
        >>> detections = detect_with_booru_yolo("path/to/image.jpg")
        >>> for box, label, confidence in detections:
        ...     print(f"Detected {label} with confidence {confidence:.2f} at {box}")
    """
    image = load_image(image, mode='RGB')
    model = _open_booru_yolo_model(model_name)
    labels = _get_booru_yolo_labels(model_name)
    new_image, old_size, new_size = _image_preprocess(image, max_infer_size)
    data = rgb_encode(new_image)[None, ...]
    output, = model.run(['output0'], {'images': data})
    return _data_postprocess(output[0], conf_threshold, iou_threshold, old_size, new_size, labels)
