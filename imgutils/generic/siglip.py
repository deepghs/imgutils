"""
SigLIP (Sigmoid Loss Image-Paired) model implementation module.

This module provides functionality for working with SigLIP models, which are designed for
image-text matching and classification tasks. It includes components for:

* Loading and managing SigLIP models from Hugging Face repositories
* Image and text encoding using ONNX models
* Prediction and classification of image-text pairs
* Web interface creation using Gradio
* Caching and thread-safe model operations

The module supports multiple model variants and provides both high-level and low-level APIs
for model interaction.
"""

import json
import os
from threading import Lock
from typing import List, Union, Optional, Any, Dict

import numpy as np
import requests
from hfutils.operate import get_hf_client, get_hf_fs
from hfutils.repository import hf_hub_repo_url
from hfutils.utils import hf_normpath, hf_fs_path, parse_hf_fs_path
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import OfflineModeIsEnabled
from tokenizers import Tokenizer

from ..data import MultiImagesTyping, load_images, ImageTyping
from ..preprocess import create_pillow_transforms
from ..utils import open_onnx_model, vreplace, sigmoid, ts_lru_cache

try:
    import gradio as gr
except (ImportError, ModuleNotFoundError):
    gr = None

__all__ = [
    'SigLIPModel',
    'siglip_image_encode',
    'siglip_text_encode',
    'siglip_predict',
]


def _check_gradio_env():
    """
    Check if the Gradio library is installed and available.

    This function verifies that Gradio is properly installed before attempting to use
    web interface features.

    :raises EnvironmentError: If Gradio is not installed, suggesting installation command.
    """
    if gr is None:
        raise EnvironmentError(f'Gradio required for launching webui-based demo.\n'
                               f'Please install it with `pip install dghs-imgutils[demo]`.')


_OFFLINE = object()


class SigLIPModel:
    """
    Main class for managing and using SigLIP models.

    This class handles model loading, caching, and inference operations for SigLIP models.
    It provides thread-safe access to model components and supports multiple model variants.

    :param repo_id: Hugging Face repository ID containing the SigLIP models
    :type repo_id: str
    :param hf_token: Optional Hugging Face authentication token
    :type hf_token: Optional[str]
    """

    def __init__(self, repo_id: str, hf_token: Optional[str] = None):
        self.repo_id = repo_id
        self._model_names = None

        self._image_encoders = {}
        self._image_preprocessors = {}
        self._text_encoders = {}
        self._text_tokenizers = {}
        self._logit_scales = {}

        self._hf_token = hf_token
        self._global_lock = Lock()
        self._model_lock = Lock()

    def _get_hf_token(self) -> Optional[str]:
        """
        Retrieve the Hugging Face authentication token.

        Checks both instance variable and environment for token presence.

        :return: Authentication token if available, None otherwise
        :rtype: Optional[str]
        """
        return self._hf_token or os.environ.get('HF_TOKEN')

    @property
    def model_names(self) -> List[str]:
        """
        Get available model names from the repository.

        This property implements lazy loading and caching of model names.
        Thread-safe access to the model list is ensured via locks.

        :return: List of available model names in the repository
        :rtype: List[str]
        :raises RuntimeError: If repository access fails
        """
        with self._global_lock:
            if self._model_names is None:
                try:
                    hf_fs = get_hf_fs(hf_token=self._get_hf_token())
                    self._model_names = [
                        hf_normpath(os.path.dirname(parse_hf_fs_path(fspath).filename))
                        for fspath in hf_fs.glob(hf_fs_path(
                            repo_id=self.repo_id,
                            repo_type='model',
                            filename='**/image_encode.onnx',
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
        Validate model name availability in the repository.

        :param model_name: Name of the model to verify
        :type model_name: str
        :raises ValueError: If model name is not found in repository
        """
        model_list = self.model_names
        if model_list is _OFFLINE:
            return  # do not check when in offline mode
        if model_name not in model_list:
            raise ValueError(f'Unknown model {model_name!r} in model repository {self.repo_id!r}, '
                             f'models {self.model_names!r} are available.')

    def _open_image_encoder(self, model_name: str):
        """
        Open and cache the ONNX image encoder model.

        :param model_name: Name of the SigLIP model variant
        :type model_name: str
        :return: Loaded ONNX model for image encoding
        :rtype: ONNXModel
        :raises ValueError: If model name is invalid
        """
        with self._model_lock:
            if model_name not in self._image_encoders:
                self._check_model_name(model_name)
                self._image_encoders[model_name] = open_onnx_model(hf_hub_download(
                    repo_id=self.repo_id,
                    repo_type='model',
                    filename=f'{model_name}/image_encode.onnx',
                ))

        return self._image_encoders[model_name]

    def _open_image_preprocessor(self, model_name: str):
        """
        Load and cache the image preprocessing pipeline configuration.

        :param model_name: Name of the SigLIP model variant
        :type model_name: str
        :return: Configured image preprocessing transforms
        :rtype: Callable
        :raises ValueError: If model name is invalid or preprocessor config is missing
        """
        with self._model_lock:
            if model_name not in self._image_preprocessors:
                self._check_model_name(model_name)
                with open(hf_hub_download(
                        repo_id=self.repo_id,
                        repo_type='model',
                        filename=f'{model_name}/preprocessor.json',
                ), 'r') as f:
                    self._image_preprocessors[model_name] = create_pillow_transforms(json.load(f)['stages'])

        return self._image_preprocessors[model_name]

    def _open_text_encoder(self, model_name: str):
        """
        Open and cache the ONNX text encoder model.

        :param model_name: Name of the SigLIP model variant
        :type model_name: str
        :return: Loaded ONNX model for text encoding
        :rtype: ONNXModel
        :raises ValueError: If model name is invalid
        """
        with self._model_lock:
            if model_name not in self._text_encoders:
                self._check_model_name(model_name)
                self._text_encoders[model_name] = open_onnx_model(hf_hub_download(
                    repo_id=self.repo_id,
                    repo_type='model',
                    filename=f'{model_name}/text_encode.onnx',
                ))

        return self._text_encoders[model_name]

    def _open_text_tokenizer(self, model_name: str):
        """
        Load and cache the text tokenizer.

        :param model_name: Name of the SigLIP model variant
        :type model_name: str
        :return: Initialized tokenizer
        :rtype: Tokenizer
        :raises ValueError: If model name is invalid or tokenizer is missing
        """
        with self._model_lock:
            if model_name not in self._text_tokenizers:
                self._check_model_name(model_name)
                self._text_tokenizers[model_name] = Tokenizer.from_file(hf_hub_download(
                    repo_id=self.repo_id,
                    repo_type='model',
                    filename=f'{model_name}/tokenizer.json',
                ))

        return self._text_tokenizers[model_name]

    def _get_logit_scale(self, model_name: str):
        """
        Get the logit scale and bias parameters from model metadata.

        :param model_name: Name of the SigLIP model variant
        :type model_name: str
        :return: Tuple of logit scale and bias values
        :rtype: tuple[float, float]
        :raises ValueError: If model name is invalid or metadata is missing
        """
        with self._model_lock:
            if model_name not in self._logit_scales:
                self._check_model_name(model_name)
                with open(hf_hub_download(
                        repo_id=self.repo_id,
                        repo_type='model',
                        filename=f'{model_name}/meta.json',
                ), 'r') as f:
                    meta_info = json.load(f)
                    self._logit_scales[model_name] = (meta_info['logit_scale'], meta_info['logit_bias'])

        return self._logit_scales[model_name]

    def _image_encode(self, images: MultiImagesTyping, model_name: str, fmt: Any = 'embeddings'):
        """
        Internal method to generate image embeddings.

        :param images: Input images to process
        :type images: MultiImagesTyping
        :param model_name: Name of the model variant to use
        :type model_name: str
        :param fmt: Output format specification
        :type fmt: Any
        :return: Image embeddings in specified format
        :raises ValueError: If model name is invalid
        """
        preprocessor = self._open_image_preprocessor(model_name)
        model = self._open_image_encoder(model_name)

        images = load_images(images, mode='RGB', force_background='white')
        input_ = np.stack([preprocessor(image) for image in images])
        encodings, embeddings = model.run(['encodings', 'embeddings'], {'pixel_values': input_})
        return vreplace(fmt, {
            'encodings': encodings,
            'embeddings': embeddings,
        })

    def image_encode(self, images: MultiImagesTyping, model_name: str, fmt: Any = 'embeddings'):
        """
        Generate embeddings for input images using the SigLIP model.

        :param images: Input images in various supported formats
        :type images: MultiImagesTyping
        :param model_name: Name of the SigLIP model variant to use
        :type model_name: str
        :param fmt: Output format, either 'encodings' or 'embeddings'
        :type fmt: Any
        :return: Image embeddings or encodings based on fmt parameter
        :raises ValueError: If model name is invalid
        """
        return self._image_encode(
            images=images,
            model_name=model_name,
            fmt=fmt,
        )

    def _text_encode(self, texts: Union[str, List[str]], model_name: str, fmt: Any = 'embeddings'):
        """
        Internal method to generate text embeddings.

        :param texts: Input text or list of texts
        :type texts: Union[str, List[str]]
        :param model_name: Name of the SigLIP model variant to use
        :type model_name: str
        :param fmt: Output format, either 'encodings' or 'embeddings'
        :type fmt: Any
        :return: Text embeddings or encodings based on fmt parameter
        :raises ValueError: If model name is invalid
        """
        tokenizer = self._open_text_tokenizer(model_name)
        model = self._open_text_encoder(model_name)

        if isinstance(texts, str):
            texts = [texts]
        encoded = tokenizer.encode_batch(texts)
        input_ids = np.stack([np.array(item.ids, dtype=np.int64) for item in encoded])
        encodings, embeddings = model.run(['encodings', 'embeddings'], {
            'input_ids': input_ids,
        })
        return vreplace(fmt, {
            'encodings': encodings,
            'embeddings': embeddings,
        })

    def text_encode(self, texts: Union[str, List[str]], model_name: str, fmt: Any = 'embeddings'):
        """
        Generate embeddings for input texts using the SigLIP model.

        :param texts: Input text or list of texts
        :type texts: Union[str, List[str]]
        :param model_name: Name of the SigLIP model variant to use
        :type model_name: str
        :param fmt: Output format, either 'encodings' or 'embeddings'
        :type fmt: Any
        :return: Text embeddings or encodings based on fmt parameter
        :raises ValueError: If model name is invalid
        """
        return self._text_encode(
            texts=texts,
            model_name=model_name,
            fmt=fmt,
        )

    def predict(
            self,
            images: Union[MultiImagesTyping, np.ndarray],
            texts: Union[List[str], str, np.ndarray],
            model_name: str,
            fmt: Any = 'predictions',
    ):
        """
        Perform image-text classification using the SigLIP model.

        :param images: Input images or pre-computed image embeddings
        :type images: Union[MultiImagesTyping, numpy.ndarray]
        :param texts: Input texts or pre-computed text embeddings
        :type texts: Union[List[str], str, numpy.ndarray]
        :param model_name: Name of the SigLIP model variant to use
        :type model_name: str
        :param fmt: Output format, one of 'similarities', 'logits', or 'predictions'
        :type fmt: Any
        :return: Classification results in specified format
        :raises ValueError: If model name is invalid
        """
        extra_values = {}
        if not isinstance(images, np.ndarray):
            image_embeddings, image_encodings = \
                self._image_encode(images, model_name=model_name, fmt=('embeddings', 'encodings'))
            extra_values['image_embeddings'] = image_embeddings
            extra_values['image_encodings'] = image_encodings
            images = image_embeddings
        images = images / np.linalg.norm(images, axis=-1, keepdims=True)

        if not isinstance(texts, np.ndarray):
            text_embeddings, text_encodings = \
                self._text_encode(texts, model_name=model_name, fmt=('embeddings', 'encodings'))
            extra_values['text_embeddings'] = text_embeddings
            extra_values['text_encodings'] = text_encodings
            texts = text_embeddings
        texts = texts / np.linalg.norm(texts, axis=-1, keepdims=True)

        similarities = images @ texts.T
        logit_scale, logit_bias = self._get_logit_scale(model_name)
        logits = similarities * np.exp(logit_scale) + logit_bias
        predictions = sigmoid(logits)

        return vreplace(fmt, {
            'similarities': similarities,
            'logits': logits,
            'predictions': predictions,
            **extra_values,
        })

    def clear(self):
        """
        Clear all cached encoders, preprocessors, tokenizers, and scales.

        This method resets the internal state of the SigLIP model by clearing all cached
        components, including image encoders, image preprocessors, text encoders,
        text tokenizers, and logit scales.
        """
        self._image_encoders.clear()
        self._image_preprocessors.clear()
        self._text_encoders.clear()
        self._text_tokenizers.clear()
        self._logit_scales.clear()

    def make_ui(self, default_model_name: Optional[str] = None):
        """
        Create an interactive Gradio UI for the SigLIP model.

        This method creates a user interface with image input, text labels input,
        model selection, and prediction display. If no default model is specified,
        it automatically selects the most recently updated model.

        :param default_model_name: Name of the model to select by default
        :type default_model_name: Optional[str]

        :raises RuntimeError: If Gradio is not properly installed
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
                    paths=[f'{model_name}/image_encode.onnx' for model_name in model_list],
                    expand=True,
            ):
                if not selected_time or fileitem.last_commit.date > selected_time:
                    selected_model_name = os.path.dirname(fileitem.path)
                    selected_time = fileitem.last_commit.date
            default_model_name = selected_model_name

        def _gr_detect(image: ImageTyping, raw_text: str, model_name: str) -> Dict[str, float]:
            labels = []
            for line in raw_text.splitlines(keepends=False):
                line = line.strip()
                if line:
                    labels.append(line)

            prediction = self.predict(images=[image], texts=labels, model_name=model_name)[0]
            return dict(zip(labels, prediction.tolist()))

        with gr.Row():
            with gr.Column():
                gr_input_image = gr.Image(type='pil', label='Original Image')
                with gr.Row():
                    gr_raw_text = gr.TextArea(value='', lines=5, autoscroll=True, label='Labels',
                                              placeholder='Enter labels, one per line')
                with gr.Row():
                    if model_list is not _OFFLINE:
                        gr_model = gr.Dropdown(model_list, value=default_model_name, label='Model')
                    else:
                        gr_model = gr.Dropdown([default_model_name], value=default_model_name, label='Model',
                                               interactive=False)

                gr_submit = gr.Button(value='Submit', variant='primary')

            with gr.Column():
                gr_output_labels = gr.Label(label='Prediction')

            gr_submit.click(
                _gr_detect,
                inputs=[
                    gr_input_image,
                    gr_raw_text,
                    gr_model,
                ],
                outputs=[gr_output_labels],
            )

    def launch_demo(self, default_model_name: Optional[str] = None,
                    server_name: Optional[str] = None, server_port: Optional[int] = None, **kwargs):
        """
        Launch a web demo for the SigLIP model.

        Creates and launches a Gradio web interface for interacting with the model.
        The demo includes the model UI and descriptive information about the model repository.

        :param default_model_name: Name of the model to select by default
        :type default_model_name: Optional[str]
        :param server_name: Server hostname to use for the demo
        :type server_name: Optional[str]
        :param server_port: Port number to use for the demo
        :type server_port: Optional[int]
        :param kwargs: Additional keyword arguments passed to gr.Blocks.launch()

        :raises RuntimeError: If Gradio is not properly installed
        """
        _check_gradio_env()
        with gr.Blocks() as demo:
            with gr.Row():
                with gr.Column():
                    repo_url = hf_hub_repo_url(repo_id=self.repo_id, repo_type='model')
                    gr.HTML(f'<h2 style="text-align: center;">SigLIP Demo For {self.repo_id}</h2>')
                    gr.Markdown(f'This is the quick demo for SigLIP model [{self.repo_id}]({repo_url}). '
                                f'Powered by `dghs-imgutils`\'s quick demo module.')

            with gr.Row():
                self.make_ui(
                    default_model_name=default_model_name,
                )

        demo.launch(
            server_name=server_name,
            server_port=server_port,
            **kwargs,
        )


@ts_lru_cache()
def _open_models_for_repo_id(repo_id: str, hf_token: Optional[str] = None) -> SigLIPModel:
    """
    Get or create a cached SigLIP model instance for the given repository ID.

    :param repo_id: Hugging Face repository ID for the model
    :type repo_id: str
    :param hf_token: Optional Hugging Face API token for private repositories
    :type hf_token: Optional[str]

    :return: A cached SigLIP model instance
    :rtype: SigLIPModel
    """
    return SigLIPModel(repo_id, hf_token=hf_token)


def siglip_image_encode(images: MultiImagesTyping, repo_id: str, model_name: str,
                        fmt: Any = 'embeddings', hf_token: Optional[str] = None):
    """
    Encode images using a SigLIP model.

    :param images: One or more images to encode
    :type images: MultiImagesTyping
    :param repo_id: Hugging Face repository ID for the model
    :type repo_id: str
    :param model_name: Name of the specific model to use
    :type model_name: str
    :param fmt: Output format ('embeddings' or custom format)
    :type fmt: Any
    :param hf_token: Optional Hugging Face API token for private repositories
    :type hf_token: Optional[str]

    :return: Encoded image features in the specified format
    :rtype: Any
    """
    model = _open_models_for_repo_id(repo_id, hf_token=hf_token)
    return model.image_encode(
        images=images,
        model_name=model_name,
        fmt=fmt,
    )


def siglip_text_encode(texts: Union[str, List[str]], repo_id: str, model_name: str,
                       fmt: Any = 'embeddings', hf_token: Optional[str] = None):
    """
    Encode texts using a SigLIP model.

    :param texts: Single text or list of texts to encode
    :type texts: Union[str, List[str]]
    :param repo_id: Hugging Face repository ID for the model
    :type repo_id: str
    :param model_name: Name of the specific model to use
    :type model_name: str
    :param fmt: Output format ('embeddings' or custom format)
    :type fmt: Any
    :param hf_token: Optional Hugging Face API token for private repositories
    :type hf_token: Optional[str]

    :return: Encoded text features in the specified format
    :rtype: Any
    """
    model = _open_models_for_repo_id(repo_id, hf_token=hf_token)
    return model.text_encode(
        texts=texts,
        model_name=model_name,
        fmt=fmt,
    )


def siglip_predict(
        images: Union[MultiImagesTyping, np.ndarray],
        texts: Union[List[str], str, np.ndarray],
        repo_id: str,
        model_name: str,
        fmt: Any = 'predictions',
        hf_token: Optional[str] = None,
):
    """
    Predict similarity scores between images and texts using a SigLIP model.

    This function computes similarity scores between the given images and texts
    using the specified SigLIP model. It can handle both raw inputs and
    pre-computed embeddings.

    :param images: Images or image embeddings to compare
    :type images: Union[MultiImagesTyping, np.ndarray]
    :param texts: Texts or text embeddings to compare
    :type texts: Union[List[str], str, np.ndarray]
    :param repo_id: Hugging Face repository ID for the model
    :type repo_id: str
    :param model_name: Name of the specific model to use
    :type model_name: str
    :param fmt: Output format ('predictions' or custom format)
    :type fmt: Any
    :param hf_token: Optional Hugging Face API token for private repositories
    :type hf_token: Optional[str]

    :return: Similarity scores in the specified format
    :rtype: Any
    """
    model = _open_models_for_repo_id(repo_id, hf_token=hf_token)
    return model.predict(
        images=images,
        texts=texts,
        model_name=model_name,
        fmt=fmt
    )
