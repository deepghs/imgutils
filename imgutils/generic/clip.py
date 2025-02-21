import json
import os
from threading import Lock
from typing import List, Union, Optional, Any

import numpy as np
from hfutils.utils import hf_normpath, parse_hf_fs_path, hf_fs_path
from huggingface_hub import hf_hub_download, HfFileSystem
from tokenizers import Tokenizer

from imgutils.data import MultiImagesTyping, load_images
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


class CLIPModel:
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
        return self._image_encode(
            images=images,
            model_name=model_name,
            fmt=fmt,
        )

    def _text_encode(self, texts: Union[str, List[str]], model_name: str, fmt: Any = 'embeddings'):
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
        self._image_encoders.clear()
        self._image_preprocessors.clear()
        self._text_encoders.clear()
        self._text_tokenizers.clear()
        self._logit_scales.clear()


@ts_lru_cache()
def _open_models_for_repo_id(repo_id: str, hf_token: Optional[str] = None) -> CLIPModel:
    return CLIPModel(repo_id, hf_token=hf_token)


def clip_image_encode(images: MultiImagesTyping, repo_id: str, model_name: str,
                      fmt='embeddings', hf_token: Optional[str] = None):
    model = _open_models_for_repo_id(repo_id, hf_token=hf_token)
    return model.image_encode(
        images=images,
        model_name=model_name,
        fmt=fmt,
    )


def clip_text_encode(texts: Union[str, List[str]], repo_id: str, model_name: str,
                     fmt='embeddings', hf_token: Optional[str] = None):
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
        fmt='predictions',
        hf_token: Optional[str] = None,
):
    model = _open_models_for_repo_id(repo_id, hf_token=hf_token)
    return model.predict(
        images=images,
        texts=texts,
        model_name=model_name,
        fmt=fmt
    )
