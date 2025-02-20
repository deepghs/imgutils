import json
import os
from threading import Lock
from typing import List, Union, Optional, Any

import numpy as np
from hfutils.utils import hf_normpath, hf_fs_path, parse_hf_fs_path
from huggingface_hub import hf_hub_download, HfFileSystem
from tokenizers import Tokenizer

from ..data import MultiImagesTyping, load_images
from ..preprocess import create_pillow_transforms
from ..utils import open_onnx_model, vreplace, sigmoid


class SigLIPModel:
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
                    hf_normpath(os.path.dirname(parse_hf_fs_path(fspath).filename))
                    for fspath in hf_fs.glob(hf_fs_path(
                        repo_id=self.repo_id,
                        repo_type='model',
                        filename='**/image_encode.onnx',
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

    def _open_image_encoder(self, model_name: str):
        """
        Open and cache the ONNX image encoder model.

        :param model_name: Name of the SigLIP model variant
        :type model_name: str
        :return: Loaded ONNX model for image encoding
        :rtype: ONNXModel
        """
        with self._model_lock:
            if model_name not in self._image_encoders:
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
        """
        with self._model_lock:
            if model_name not in self._image_preprocessors:
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
        """
        with self._model_lock:
            if model_name not in self._text_encoders:
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
        """
        with self._model_lock:
            if model_name not in self._text_tokenizers:
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
        """
        with self._model_lock:
            if model_name not in self._logit_scales:
                with open(hf_hub_download(
                        repo_id=self.repo_id,
                        repo_type='model',
                        filename=f'{model_name}/meta.json',
                ), 'r') as f:
                    meta_info = json.load(f)
                    self._logit_scales[model_name] = (meta_info['logit_scale'], meta_info['logit_bias'])

        return self._logit_scales[model_name]

    def get_siglip_image_embedding(self, images: MultiImagesTyping, model_name: str, fmt: Any = 'embeddings'):
        """
        Generate embeddings for input images using the SigLIP model.
    
        :param images: Input images in various supported formats
        :type images: MultiImagesTyping
        :param model_name: Name of the SigLIP model variant to use
        :type model_name: str
        :param fmt: Output format, either 'encodings' or 'embeddings'
    
        :return: Image embeddings or encodings based on fmt parameter
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

    def get_siglip_text_embedding(self, texts: Union[str, List[str]], model_name: str, fmt: Any = 'embeddings'):
        """
        Generate embeddings for input texts using the SigLIP model.
    
        :param texts: Input text or list of texts
        :type texts: Union[str, List[str]]
        :param model_name: Name of the SigLIP model variant to use
        :type model_name: str
        :param fmt: Output format, either 'encodings' or 'embeddings'
    
        :return: Text embeddings or encodings based on fmt parameter
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

    def classify_with_siglip(
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

        :return: Classification results in specified format
        """
        extra_values = {}
        if not isinstance(images, np.ndarray):
            image_embeddings, image_encodings = \
                self.get_siglip_image_embedding(images, model_name=model_name, fmt=('embeddings', 'encodings'))
            extra_values['image_embeddings'] = image_embeddings
            extra_values['image_encodings'] = image_encodings
            images = image_embeddings
        images = images / np.linalg.norm(images, axis=-1, keepdims=True)

        if not isinstance(texts, np.ndarray):
            text_embeddings, text_encodings = \
                self.get_siglip_text_embedding(texts, model_name=model_name, fmt=('embeddings', 'encodings'))
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
