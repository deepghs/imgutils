"""
CLIP model interface for multimodal embeddings and predictions.

This module provides a comprehensive interface for working with CLIP models hosted on Hugging Face Hub.

The main class `CLIPModel` handles model management and provides:

- Automatic discovery of available model variants
- ONNX runtime integration for efficient inference
- Preprocessing pipelines for images and text
- Similarity calculation and prediction methods

Typical usage patterns:

1. Direct API usage through clip_image_encode/clip_text_encode/clip_predict functions
2. Instance-based control via CLIPModel class
3. Web demo deployment through launch_demo method

.. note::
    For optimal performance with multiple models, reuse CLIPModel instances when possible.
    The module implements LRU caching for model instances based on repository ID.
"""

import json
import os
from threading import Lock
from typing import List, Union, Optional, Any, Dict

import numpy as np
import requests
from hfutils.operate import get_hf_client, get_hf_fs
from hfutils.repository import hf_hub_repo_url
from hfutils.utils import hf_normpath, parse_hf_fs_path, hf_fs_path
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import OfflineModeIsEnabled
from tokenizers import Tokenizer

from imgutils.data import MultiImagesTyping, load_images, ImageTyping
from imgutils.preprocess import create_pillow_transforms
from imgutils.utils import open_onnx_model, vreplace, ts_lru_cache

try:
    import gradio as gr
except (ImportError, ModuleNotFoundError):
    gr = None

__all__ = [
    'CLIPModel',
    'clip_image_encode',
    'clip_text_encode',
    'clip_predict',
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


class CLIPModel:
    """
    Main interface for CLIP model operations.

    This class provides thread-safe access to CLIP model variants stored in a Hugging Face repository.
    It handles model loading, preprocessing, inference, and provides web interface capabilities.

    :param repo_id: Hugging Face repository ID containing CLIP models
    :type repo_id: str
    :param hf_token: Optional authentication token for private repositories
    :type hf_token: Optional[str]

    .. note::
        Model components are loaded on-demand and cached for subsequent use.
        Use clear() method to free memory when working with multiple large models.
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

        .. note::
            Model names are discovered by searching for 'image_encode.onnx' files
            in the repository's directory structure.
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
        Load and cache image encoder ONNX model.

        :param model_name: Target model variant name
        :return: Loaded ONNX model session
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
        Load and cache image preprocessing transforms.

        :param model_name: Target model variant name
        :return: Preprocessing pipeline function
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
        Load and cache text encoder ONNX model.

        :param model_name: Target model variant name
        :return: Loaded ONNX model session
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
        Load and cache text tokenizer.

        :param model_name: Target model variant name
        :return: Configured tokenizer instance
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
        Get and cache the logit scale factor for the model.

        :param model_name: Name of the CLIP model variant
        :type model_name: str

        :return: Logit scale value
        :rtype: float
        """
        with self._model_lock:
            if model_name not in self._logit_scales:
                self._check_model_name(model_name)
                with open(hf_hub_download(
                        repo_id=self.repo_id,
                        repo_type='model',
                        filename=f'{model_name}/meta.json',
                ), 'r') as f:
                    self._logit_scales[model_name] = json.load(f)['logit_scale']

        return self._logit_scales[model_name]

    def _image_encode(self, images: MultiImagesTyping, model_name: str, fmt: Any = 'embeddings'):
        """
        Internal implementation of image encoding.

        :param images: Input images to encode
        :param model_name: Target model variant
        :param fmt: Output format specification
        :return: Encoded image features in specified format
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
        Encode images into CLIP embeddings.

        :param images: Input images (paths, URLs, PIL images, or numpy arrays)
        :type images: MultiImagesTyping
        :param model_name: Target model variant name
        :type model_name: str
        :param fmt: Output format specification. Can be:
            - 'embeddings' (default): Return normalized embeddings
            - 'encodings': Return raw model outputs
            - Tuple of both: Return (embeddings, encodings)
        :type fmt: Any

        :return: Encoded features in specified format
        :rtype: Any

        .. note::
            Input images are automatically converted to RGB format with white background.
        """
        return self._image_encode(
            images=images,
            model_name=model_name,
            fmt=fmt,
        )

    def _text_encode(self, texts: Union[str, List[str]], model_name: str, fmt: Any = 'embeddings'):
        """
        Internal implementation of text encoding.

        :param texts: Input texts to encode
        :param model_name: Target model variant
        :param fmt: Output format specification
        :return: Encoded text features in specified format
        """
        tokenizer = self._open_text_tokenizer(model_name)
        model = self._open_text_encoder(model_name)

        if isinstance(texts, str):
            texts = [texts]
        encoded = tokenizer.encode_batch(texts)
        input_ids = np.stack([np.array(item.ids, dtype=np.int64) for item in encoded])
        attention_mask = np.stack([np.array(item.attention_mask, dtype=np.int64) for item in encoded])
        encodings, embeddings = model.run(['encodings', 'embeddings'], {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        })
        return vreplace(fmt, {
            'encodings': encodings,
            'embeddings': embeddings,
        })

    def text_encode(self, texts: Union[str, List[str]], model_name: str, fmt: Any = 'embeddings'):
        """
        Encode text into CLIP embeddings.

        :param texts: Input text or list of texts
        :type texts: Union[str, List[str]]
        :param model_name: Target model variant name
        :type model_name: str
        :param fmt: Output format specification. Can be:
            - 'embeddings' (default): Return normalized embeddings
            - 'encodings': Return raw model outputs
            - Tuple of both: Return (embeddings, encodings)
        :type fmt: Any

        :return: Encoded features in specified format
        :rtype: Any
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
            fmt='predictions'
    ):
        """
        Calculate similarity predictions between images and texts.

        :param images: Input images or precomputed embeddings
        :type images: Union[MultiImagesTyping, np.ndarray]
        :param texts: Input texts or precomputed embeddings
        :type texts: Union[List[str], str, np.ndarray]
        :param model_name: Target model variant name
        :type model_name: str
        :param fmt: Output format specification. Can be:
            - 'predictions' (default): Normalized probability scores
            - 'similarities': Cosine similarities
            - 'logits': Scaled similarity scores
            - Complex format using dict keys:
                ('image_embeddings', 'text_embeddings', 'similarities', etc.)
        :type fmt: Any

        :return: Prediction results in specified format
        :rtype: Any

        .. note::
            When passing precomputed embeddings, ensure they are L2-normalized
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
        logits = similarities * np.exp(self._get_logit_scale(model_name=model_name))
        predictions = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)

        return vreplace(fmt, {
            'similarities': similarities,
            'logits': logits,
            'predictions': predictions,
            **extra_values,
        })

    def clear(self):
        """
        Clear all cached models and components.

        Use this to free memory when switching between different model variants.
        """
        self._model_names = None
        self._image_encoders.clear()
        self._image_preprocessors.clear()
        self._text_encoders.clear()
        self._text_tokenizers.clear()
        self._logit_scales.clear()

    def make_ui(self, default_model_name: Optional[str] = None):
        """
        Create Gradio interface components for an interactive CLIP model demo.

        This method sets up a user interface with image input, text input for labels,
        model selection dropdown, and prediction display. It automatically selects the
        most recently updated model variant if no default is specified.

        :param default_model_name: Optional name of the model variant to select by default.
                                 If None, the most recently updated model variant will be selected.
        :type default_model_name: Optional[str]
        :return: None
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
        Launch a Gradio web interface for interactive CLIP model predictions.

        Creates and launches a web demo that allows users to upload images, enter text labels,
        and get similarity predictions using the CLIP model. The interface includes model
        information and repository links.

        :param default_model_name: Initial model variant to select in the dropdown
        :type default_model_name: Optional[str]
        :param server_name: Host address to bind the server to (e.g., "0.0.0.0" for public access)
        :type server_name: Optional[str]
        :param server_port: Port number to run the server on
        :type server_port: Optional[int]
        :param kwargs: Additional keyword arguments passed to gradio.launch()
        :return: None

        Usage:
            >>> model = CLIPModel("organization/model-name")
            >>> model.launch_demo(server_name="0.0.0.0", server_port=7860)
        """
        _check_gradio_env()
        with gr.Blocks() as demo:
            with gr.Row():
                with gr.Column():
                    repo_url = hf_hub_repo_url(repo_id=self.repo_id, repo_type='model')
                    gr.HTML(f'<h2 style="text-align: center;">CLIP Demo For {self.repo_id}</h2>')
                    gr.Markdown(f'This is the quick demo for CLIP model [{self.repo_id}]({repo_url}). '
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
def _open_models_for_repo_id(repo_id: str, hf_token: Optional[str] = None) -> CLIPModel:
    """
    Load and cache a CLIP model instance for the given repository ID.

    This function uses a thread-safe LRU cache to avoid repeatedly loading the same model.

    :param repo_id: Hugging Face model repository ID
    :type repo_id: str
    :param hf_token: Optional Hugging Face API token for private models
    :type hf_token: Optional[str]
    :return: Cached CLIP model instance
    :rtype: CLIPModel
    """
    return CLIPModel(repo_id, hf_token=hf_token)


def clip_image_encode(images: MultiImagesTyping, repo_id: str, model_name: str,
                      fmt: Any = 'embeddings', hf_token: Optional[str] = None):
    """
    Generate CLIP embeddings or features for the given images.

    :param images: Input images (paths, PIL Images, or numpy arrays)
    :type images: MultiImagesTyping
    :param repo_id: Hugging Face model repository ID
    :type repo_id: str
    :param model_name: Name of the specific model variant to use
    :type model_name: str
    :param fmt: Output format ('embeddings' or 'features')
    :type fmt: Any
    :param hf_token: Optional Hugging Face API token
    :type hf_token: Optional[str]
    :return: Image embeddings or features
    """
    model = _open_models_for_repo_id(repo_id, hf_token=hf_token)
    return model.image_encode(
        images=images,
        model_name=model_name,
        fmt=fmt,
    )


def clip_text_encode(texts: Union[str, List[str]], repo_id: str, model_name: str,
                     fmt: Any = 'embeddings', hf_token: Optional[str] = None):
    """
    Generate CLIP embeddings or features for the given texts.

    :param texts: Input text or list of texts
    :type texts: Union[str, List[str]]
    :param repo_id: Hugging Face model repository ID
    :type repo_id: str
    :param model_name: Name of the specific model variant to use
    :type model_name: str
    :param fmt: Output format ('embeddings' or 'features')
    :type fmt: Any
    :param hf_token: Optional Hugging Face API token
    :type hf_token: Optional[str]
    :return: Text embeddings or features
    """
    model = _open_models_for_repo_id(repo_id, hf_token=hf_token)
    return model.text_encode(
        texts=texts,
        model_name=model_name,
        fmt=fmt,
    )


def clip_predict(
        images: Union[MultiImagesTyping, np.ndarray],
        texts: Union[List[str], str, np.ndarray],
        repo_id: str,
        model_name: str,
        fmt: Any = 'predictions',
        hf_token: Optional[str] = None,
):
    """
    Calculate similarity scores between images and texts using CLIP.

    This function computes the similarity between the given images and texts
    using the specified CLIP model. It can accept raw images/texts or
    pre-computed embeddings as input.

    :param images: Input images or pre-computed image embeddings
    :type images: Union[MultiImagesTyping, np.ndarray]
    :param texts: Input texts or pre-computed text embeddings
    :type texts: Union[List[str], str, np.ndarray]
    :param repo_id: Hugging Face model repository ID
    :type repo_id: str
    :param model_name: Name of the specific model variant to use
    :type model_name: str
    :param fmt: Output format ('predictions' for similarity scores or 'logits' for raw logits)
    :type fmt: Any
    :param hf_token: Optional Hugging Face API token
    :type hf_token: Optional[str]
    :return: Similarity scores or logits between images and texts
    """
    model = _open_models_for_repo_id(repo_id, hf_token=hf_token)
    return model.predict(
        images=images,
        texts=texts,
        model_name=model_name,
        fmt=fmt
    )
