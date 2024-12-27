"""
Generic tools for classification models.

This module provides utilities and classes for working with classification models,
particularly those stored in Hugging Face repositories. It includes functions for
image encoding, model loading, and prediction, as well as a main ClassifyModel class
that manages the interaction with classification models.

The module is designed to work with ONNX models and supports various image input formats.
It also handles token-based authentication for accessing private Hugging Face repositories.
"""

import json
import os
from threading import Lock
from typing import Tuple, Optional, List, Dict, Callable

import numpy as np
from PIL import Image
from hfutils.operate import get_hf_client
from hfutils.repository import hf_hub_repo_url
from hfutils.utils import hf_fs_path, hf_normpath
from huggingface_hub import hf_hub_download, HfFileSystem

from ..data import rgb_encode, ImageTyping, load_image
from ..utils import open_onnx_model, ts_lru_cache

try:
    import gradio as gr
except (ImportError, ModuleNotFoundError):
    gr = None

__all__ = [
    'ClassifyModel',
    'classify_predict_score',
    'classify_predict',
]


def _check_gradio_env():
    """
    Verify that Gradio library is properly installed and available.

    This function checks if the Gradio package is accessible for creating
    web-based demos. If Gradio is not found, it provides instructions for installation.

    :raises EnvironmentError: If Gradio package is not installed in the environment.
    """
    if gr is None:
        raise EnvironmentError(f'Gradio required for launching webui-based demo.\n'
                               f'Please install it with `pip install dghs-imgutils[demo]`.')


def _img_encode(image: Image.Image, size: Tuple[int, int] = (384, 384),
                normalize: Optional[Tuple[float, float]] = (0.5, 0.5)):
    """
    Encode an image into a numpy array suitable for model input.

    This function performs several preprocessing steps on the input image:
        1. Resizes the image to the specified dimensions
        2. Converts to RGB format
        3. Applies normalization if parameters are provided
        4. Returns the image in CHW (Channel, Height, Width) format

    :param image: Input PIL Image to be encoded
    :type image: Image.Image
    :param size: Target dimensions (width, height) for resizing, defaults to (384, 384)
    :type size: Tuple[int, int]
    :param normalize: Normalization parameters (mean, std), defaults to (0.5, 0.5)
    :type normalize: Optional[Tuple[float, float]]

    :return: Encoded and preprocessed image as numpy array
    :rtype: np.ndarray

    :raises TypeError: If input is not a PIL Image

    Example:
        >>> img = Image.open('example.jpg')
        >>> encoded = _img_encode(img, size=(224, 224))
    """
    image = image.resize(size, Image.BILINEAR)
    data = rgb_encode(image, order_='CHW')

    if normalize is not None:
        mean_, std_ = normalize
        mean = np.asarray([mean_]).reshape((-1, 1, 1))
        std = np.asarray([std_]).reshape((-1, 1, 1))
        data = (data - mean) / std

    return data.astype(np.float32)


ImagePreprocessFunc = Callable[[Image.Image], Image.Image]


class ClassifyModel:
    """
    A comprehensive manager for classification models from Hugging Face repositories.

    :param repo_id: Hugging Face repository identifier
    :type repo_id: str
    :param fn_preprocess: Optional custom preprocessing function
    :type fn_preprocess: Optional[ImagePreprocessFunc]
    :param hf_token: Hugging Face authentication token
    :type hf_token: Optional[str]

    :ivar repo_id: Repository identifier
    :ivar _model_names: Cached list of available models
    :ivar _models: Dictionary of loaded ONNX models
    :ivar _labels: Dictionary of model labels
    :ivar _hf_token: Authentication token

    Usage:
        >>> classifier = ClassifyModel("org/model-repo")
        >>> with Image.open("image.jpg") as img:
        ...     label = classifier.predict(img, "model-name")
    """

    def __init__(self, repo_id: str, fn_preprocess: Optional[ImagePreprocessFunc] = None,
                 hf_token: Optional[str] = None):
        """
        Initialize a new ClassifyModel instance.

        :param repo_id: Hugging Face repository identifier
        :type repo_id: str
        :param fn_preprocess: Optional custom preprocessing function
        :type fn_preprocess: Optional[ImagePreprocessFunc]
        :param hf_token: Authentication token for private repositories
        :type hf_token: Optional[str]
        """
        self.repo_id = repo_id
        self._fn_preprocess = fn_preprocess
        self._model_names = None
        self._models = {}
        self._labels = {}
        self._hf_token = hf_token
        self._global_lock = Lock()
        self._model_lock = Lock()

    def _get_hf_token(self) -> Optional[str]:
        """
        Retrieve the Hugging Face authentication token.

        Checks both instance variable and environment for token presence.

        :return: Authentication token if available
        :rtype: Optional[str]
        """
        return self._hf_token or os.environ.get('HF_TOKEN')

    @property
    def model_names(self) -> List[str]:
        """
        Get available model names from the repository.

        This property implements lazy loading and caching of model names.
        Thread-safe access to the model list is ensured via locks.

        :return: List of available model names
        :rtype: List[str]

        :raises RuntimeError: If repository access fails
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
        Validate model name availability in the repository.

        :param model_name: Name of the model to verify
        :type model_name: str

        :raises ValueError: If model name is not found in repository
        """
        if model_name not in self.model_names:
            raise ValueError(f'Unknown model {model_name!r} in model repository {self.repo_id!r}, '
                             f'models {self.model_names!r} are available.')

    def _open_model(self, model_name: str):
        """
        Load and cache an ONNX model.

        Implements thread-safe model loading with caching for improved performance.
        Downloads model from Hugging Face if not locally available.

        :param model_name: Name of the model to load
        :type model_name: str

        :return: Loaded ONNX model
        :rtype: Any

        :raises RuntimeError: If model loading fails
        """
        with self._model_lock:
            if model_name not in self._models:
                # read HF_HUB_OFFLINE in environment variables
                # when HF_HUB_OFFLINE=1, the model is skipped and the locally cached model is loaded
                if not os.getenv('HF_HUB_OFFLINE') == "1":
                    self._check_model_name(model_name)
                # if there is no local cache model, and HF_HUB_OFFLINE=1, an exception is thrown
                try:
                    self._models[model_name] = open_onnx_model(hf_hub_download(
                        self.repo_id,
                        f'{model_name}/model.onnx',
                        token=self._get_hf_token(),
                    ))
                except Exception as e:
                    if os.getenv('HF_HUB_OFFLINE') == "1":
                        raise Exception(
                            "You have turned on environment variables, HF_HUB_OFFLINE=1, but there are no cache files locally, please unset HF_HUB_OFFLINE=1 and enable it after downloading the model")
                    else:
                        raise Exception(e)

        return self._models[model_name]

    def _open_label(self, model_name: str) -> List[str]:
        """
        Load and cache model labels from metadata.

        Implements thread-safe loading of model labels with caching.
        Downloads label metadata from Hugging Face if not locally available.

        :param model_name: Name of the model whose labels to load
        :type model_name: str

        :return: List of model labels
        :rtype: List[str]

        :raises RuntimeError: If label loading fails
        """
        with self._model_lock:
            if model_name not in self._labels:
                self._check_model_name(model_name)
                with open(hf_hub_download(
                        self.repo_id,
                        f'{model_name}/meta.json',
                        token=self._get_hf_token(),
                ), 'r') as f:
                    self._labels[model_name] = json.load(f)['labels']

        return self._labels[model_name]

    def _raw_predict(self, image: ImageTyping, model_name: str):
        """
        Generate raw model predictions for an input image.

        This method handles:
            1. Image loading and preprocessing
            2. Model input shape validation
            3. Custom preprocessing if specified
            4. Model inference

        :param image: Input image for prediction
        :type image: ImageTyping
        :param model_name: Name of model to use
        :type model_name: str

        :return: Raw model output
        :rtype: np.ndarray

        :raises RuntimeError: If model input shape is incompatible
        """
        image = load_image(image, force_background='white', mode='RGB')
        model = self._open_model(model_name)
        batch, channels, height, width = model.get_inputs()[0].shape
        if channels != 3:
            raise RuntimeError(f'Model {model_name!r} required {[batch, channels, height, width]!r}, '
                               f'channels not 3.')  # pragma: no cover

        if self._fn_preprocess:
            image = self._fn_preprocess(image)

        if isinstance(height, int) and isinstance(width, int):
            input_ = _img_encode(image, size=(width, height))[None, ...]
        else:
            input_ = _img_encode(image)[None, ...]
        output, = self._open_model(model_name).run(['output'], {'input': input_})
        return output

    def predict_score(self, image: ImageTyping, model_name: str) -> Dict[str, float]:
        """
        Predict the scores for each class using the specified model.

        This method runs the image through the model and returns a dictionary of class scores.

        :param image: The input image to classify.
        :type image: ImageTyping
        :param model_name: The name of the model to use for prediction.
        :type model_name: str

        :return: A dictionary mapping class labels to their predicted scores.
        :rtype: Dict[str, float]

        :raises ValueError: If the model name is invalid.
        :raises RuntimeError: If there's an error during prediction.
        """
        output = self._raw_predict(image, model_name)
        values = dict(zip(self._open_label(model_name), map(lambda x: x.item(), output[0])))
        return values

    def predict(self, image: ImageTyping, model_name: str) -> Tuple[str, float]:
        """
        Predict the class with the highest score for the given image.

        This method runs the image through the model and returns the predicted class and its score.

        :param image: The input image to classify.
        :type image: ImageTyping
        :param model_name: The name of the model to use for prediction.
        :type model_name: str

        :return: A tuple containing the predicted class label and its score.
        :rtype: Tuple[str, float]

        :raises ValueError: If the model name is invalid.
        :raises RuntimeError: If there's an error during prediction.
        """
        output = self._raw_predict(image, model_name)[0]
        max_id = np.argmax(output)
        return self._open_label(model_name)[max_id], output[max_id].item()

    def clear(self):
        """
        Clear the cached models and labels.

        This method frees up memory by removing all loaded models and labels from the cache.
        """
        self._models.clear()
        self._labels.clear()

    def make_ui(self, default_model_name: Optional[str] = None):
        """
        Create the user interface components for the classifier model demo.

        This method sets up the Gradio UI components including an image input, model selection dropdown,
        submit button, and output label. It also configures the interaction between these components.

        :param default_model_name: The name of the default model to be selected in the dropdown.
                                   If None, the most recently updated model will be selected.
        :type default_model_name: Optional[str]

        :raises ImportError: If Gradio is not installed or properly configured.

        :Example:
        >>> model = ClassifyModel("username/repo_name")
        >>> model.make_ui(default_model_name="model_v1")
        """

        # demo for classifier model
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

        with gr.Row():
            with gr.Column():
                gr_input_image = gr.Image(type='pil', label='Original Image')
                gr_model = gr.Dropdown(model_list, value=default_model_name, label='Model')
                gr_submit = gr.Button(value='Submit', variant='primary')

            with gr.Column():
                gr_output = gr.Label(label='Prediction')

            gr_submit.click(
                self.predict_score,
                inputs=[
                    gr_input_image,
                    gr_model,
                ],
                outputs=[gr_output],
            )

    def launch_demo(self, default_model_name: Optional[str] = None,
                    server_name: Optional[str] = None, server_port: Optional[int] = None, **kwargs):
        """
        Launch the Gradio demo for the classifier model.

        This method creates a Gradio Blocks interface, sets up the UI components using make_ui(),
        and launches the demo server.

        :param default_model_name: The name of the default model to be selected in the dropdown.
        :type default_model_name: Optional[str]
        :param server_name: The name of the server to run the demo on. Defaults to None.
        :type server_name: Optional[str]
        :param server_port: The port number to run the demo on. Defaults to None.
        :type server_port: Optional[int]
        :param kwargs: Additional keyword arguments to pass to the Gradio launch method.

        :raises ImportError: If Gradio is not installed or properly configured.

        :Example:
        >>> model = ClassifyModel("username/repo_name")
        >>> model.launch_demo(default_model_name="model_v1", server_name="0.0.0.0", server_port=7860)
        """

        _check_gradio_env()
        with gr.Blocks() as demo:
            with gr.Row():
                with gr.Column():
                    repo_url = hf_hub_repo_url(repo_id=self.repo_id, repo_type='model')
                    gr.HTML(f'<h2 style="text-align: center;">Classifier Demo For {self.repo_id}</h2>')
                    gr.Markdown(f'This is the quick demo for classifier model [{self.repo_id}]({repo_url}). '
                                f'Powered by `dghs-imgutils`\'s quick demo module.')

            with gr.Row():
                self.make_ui(default_model_name=default_model_name)

        demo.launch(
            server_name=server_name,
            server_port=server_port,
            **kwargs,
        )


@ts_lru_cache()
def _open_models_for_repo_id(repo_id: str, hf_token: Optional[str] = None) -> ClassifyModel:
    """
    Open and cache a ClassifyModel instance for the specified repository ID.

    This function uses LRU caching to avoid creating multiple ClassifyModel instances
    for the same repository.

    :param repo_id: The repository ID containing the models.
    :type repo_id: str
    :param hf_token: Optional Hugging Face authentication token.
    :type hf_token: Optional[str]

    :return: A ClassifyModel instance for the specified repository.
    :rtype: ClassifyModel
    """
    return ClassifyModel(repo_id, hf_token=hf_token)


def classify_predict_score(image: ImageTyping, repo_id: str, model_name: str,
                           hf_token: Optional[str] = None) -> Dict[str, float]:
    """
    Predict the scores for each class using the specified model and repository.

    This function is a convenience wrapper around ClassifyModel's predict_score method.

    :param image: The input image to classify.
    :type image: ImageTyping
    :param repo_id: The repository ID containing the models.
    :type repo_id: str
    :param model_name: The name of the model to use for prediction.
    :type model_name: str
    :param hf_token: Optional Hugging Face authentication token.
    :type hf_token: Optional[str]

    :return: A dictionary mapping class labels to their predicted scores.
    :rtype: Dict[str, float]

    :raises ValueError: If the model name or repository ID is invalid.
    :raises RuntimeError: If there's an error during prediction.
    """
    return _open_models_for_repo_id(repo_id, hf_token=hf_token).predict_score(image, model_name)


def classify_predict(image: ImageTyping, repo_id: str, model_name: str,
                     hf_token: Optional[str] = None) -> Tuple[str, float]:
    """
    Predict the class with the highest score using the specified model and repository.

    This function is a convenience wrapper around ClassifyModel's predict method.

    :param image: The input image to classify.
    :type image: ImageTyping
    :param repo_id: The repository ID containing the models.
    :type repo_id: str
    :param model_name: The name of the model to use for prediction.
    :type model_name: str
    :param hf_token: Optional Hugging Face authentication token.
    :type hf_token: Optional[str]

    :return: A tuple containing the predicted class label and its score.
    :rtype: Tuple[str, float]

    :raises ValueError: If the model name or repository ID is invalid.
    :raises RuntimeError: If there's an error during prediction.
    """
    return _open_models_for_repo_id(repo_id, hf_token=hf_token).predict(image, model_name)
