"""
Generic tools for classification models.

This module provides utilities and classes for working with classification models,
particularly those stored in Hugging Face repositories. It includes functions for
image encoding, model loading, and prediction, as well as a main `ClassifyModel` class
that manages the interaction with classification models.

Key components:

- Image encoding and preprocessing
- ClassifyModel: A class for managing and using classification models
- Utility functions for making predictions with classification models

The module is designed to work with ONNX models and supports various image input formats.
It also handles token-based authentication for accessing private Hugging Face repositories.
"""

import json
import os
from functools import lru_cache
from typing import Tuple, Optional, List, Dict

import numpy as np
from PIL import Image
from hfutils.utils import hf_fs_path
from huggingface_hub import hf_hub_download, HfFileSystem

from ..data import rgb_encode, ImageTyping, load_image
from ..utils import open_onnx_model

__all__ = [
    'ClassifyModel',
    'classify_predict_score',
    'classify_predict',
]


def _img_encode(image: Image.Image, size: Tuple[int, int] = (384, 384),
                normalize: Optional[Tuple[float, float]] = (0.5, 0.5)):
    """
    Encode an image into a numpy array for model input.

    This function resizes the input image, converts it to RGB format, and optionally
    normalizes the pixel values.

    :param image: The input image to be encoded.
    :type image: Image.Image
    :param size: The target size (width, height) to resize the image to, defaults to (384, 384).
    :type size: Tuple[int, int], optional
    :param normalize: The mean and standard deviation for normalization, defaults to (0.5, 0.5).
                      If None, no normalization is applied.
    :type normalize: Optional[Tuple[float, float]], optional

    :return: The encoded image as a numpy array in CHW format.
    :rtype: np.ndarray

    :raises TypeError: If the input image is not a PIL Image object.
    """
    image = image.resize(size, Image.BILINEAR)
    data = rgb_encode(image, order_='CHW')

    if normalize is not None:
        mean_, std_ = normalize
        mean = np.asarray([mean_]).reshape((-1, 1, 1))
        std = np.asarray([std_]).reshape((-1, 1, 1))
        data = (data - mean) / std

    return data.astype(np.float32)


class ClassifyModel:
    """
    A class for managing and using classification models.

    This class provides methods for loading classification models from a Hugging Face
    repository, making predictions, and managing model resources. It supports multiple
    models within a single repository and handles token-based authentication.

    :param repo_id: The Hugging Face repository ID containing the classification models.
    :type repo_id: str
    :param hf_token: The Hugging Face API token for accessing private repositories, defaults to None.
    :type hf_token: Optional[str], optional

    :ivar repo_id: The Hugging Face repository ID.
    :ivar _model_names: Cached list of available model names in the repository.
    :ivar _models: Dictionary of loaded ONNX models.
    :ivar _labels: Dictionary of labels for each model.
    :ivar _hf_token: The Hugging Face API token.

    Usage:
        >>> model = ClassifyModel("username/repo_name")
        >>> image = Image.open("path/to/image.jpg")
        >>> prediction, score = model.predict(image, "model_name")
        >>> print(f"Predicted class: {prediction}, Score: {score}")
    """

    def __init__(self, repo_id: str, hf_token: Optional[str] = None):
        """
        Initialize the ClassifyModel instance.

        :param repo_id: The repository ID containing the models.
        :type repo_id: str
        :param hf_token: The Hugging Face API token, defaults to None.
        :type hf_token: Optional[str], optional
        """
        self.repo_id = repo_id
        self._model_names = None
        self._models = {}
        self._labels = {}
        self._hf_token = hf_token

    def _get_hf_token(self):
        """
        Get the Hugging Face token from the instance variable or environment variable.

        :return: The Hugging Face token.
        :rtype: str
        """
        return self._hf_token or os.environ.get('HF_TOKEN')

    @property
    def model_names(self) -> List[str]:
        """
        Get the list of available model names in the repository.

        This property lazily loads the model names from the Hugging Face repository
        and caches them for future use.

        :return: The list of model names available in the repository.
        :rtype: List[str]

        :raises RuntimeError: If there's an error accessing the Hugging Face repository.
        """
        if self._model_names is None:
            hf_fs = HfFileSystem(token=self._get_hf_token())
            self._model_names = [
                os.path.dirname(os.path.relpath(item, self.repo_id))
                for item in hf_fs.glob(hf_fs_path(
                    repo_id=self.repo_id,
                    repo_type='model',
                    filename='*/model.onnx',
                ))
            ]

        return self._model_names

    def _check_model_name(self, model_name: str):
        """
        Check if the given model name is valid and available in the repository.

        :param model_name: The name of the model to check.
        :type model_name: str

        :raises ValueError: If the model name is not found in the list of available models.
        """
        if model_name not in self.model_names:
            raise ValueError(f'Unknown model {model_name!r} in model repository {self.repo_id!r}, '
                             f'models {self.model_names!r} are available.')

    def _open_model(self, model_name: str):
        """
        Open and cache the specified ONNX model.

        This method downloads the model if it's not already cached and opens it using ONNX runtime.

        :param model_name: The name of the model to open.
        :type model_name: str

        :return: The opened ONNX model.
        :rtype: Any

        :raises RuntimeError: If there's an error downloading or opening the model.
        """
        if model_name not in self._models:
            self._check_model_name(model_name)
            self._models[model_name] = open_onnx_model(hf_hub_download(
                self.repo_id,
                f'{model_name}/model.onnx',
                token=self._get_hf_token(),
            ))
        return self._models[model_name]

    def _open_label(self, model_name: str) -> List[str]:
        """
        Open and cache the labels file for the specified model.

        This method downloads the meta.json file containing the labels if it's not already cached.

        :param model_name: The name of the model whose labels to open.
        :type model_name: str

        :return: The list of labels for the specified model.
        :rtype: List[str]

        :raises RuntimeError: If there's an error downloading or parsing the labels file.
        """
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
        Make a raw prediction on the specified image using the specified model.

        This method preprocesses the image, runs it through the model, and returns the raw output.

        :param image: The input image to classify.
        :type image: ImageTyping
        :param model_name: The name of the model to use for prediction.
        :type model_name: str

        :return: The raw prediction output from the model.
        :rtype: np.ndarray

        :raises RuntimeError: If the model's input shape is incompatible with the image.
        """
        image = load_image(image, force_background='white', mode='RGB')
        model = self._open_model(model_name)
        batch, channels, height, width = model.get_inputs()[0].shape
        if channels != 3:
            raise RuntimeError(f'Model {model_name!r} required {[batch, channels, height, width]!r}, '
                               f'channels not 3.')  # pragma: no cover

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


@lru_cache()
def _open_models_for_repo_id(repo_id: str) -> ClassifyModel:
    """
    Open and cache a ClassifyModel instance for the specified repository ID.

    This function uses LRU caching to avoid creating multiple ClassifyModel instances
    for the same repository.

    :param repo_id: The repository ID containing the models.
    :type repo_id: str

    :return: A ClassifyModel instance for the specified repository.
    :rtype: ClassifyModel
    """
    return ClassifyModel(repo_id)


def classify_predict_score(image: ImageTyping, repo_id: str, model_name: str) -> Dict[str, float]:
    """
    Predict the scores for each class using the specified model and repository.

    This function is a convenience wrapper around ClassifyModel's predict_score method.

    :param image: The input image to classify.
    :type image: ImageTyping
    :param repo_id: The repository ID containing the models.
    :type repo_id: str
    :param model_name: The name of the model to use for prediction.
    :type model_name: str

    :return: A dictionary mapping class labels to their predicted scores.
    :rtype: Dict[str, float]

    :raises ValueError: If the model name or repository ID is invalid.
    :raises RuntimeError: If there's an error during prediction.
    """
    return _open_models_for_repo_id(repo_id).predict_score(image, model_name)


def classify_predict(image: ImageTyping, repo_id: str, model_name: str) -> Tuple[str, float]:
    """
    Predict the class with the highest score using the specified model and repository.

    This function is a convenience wrapper around ClassifyModel's predict method.

    :param image: The input image to classify.
    :type image: ImageTyping
    :param repo_id: The repository ID containing the models.
    :type repo_id: str
    :param model_name: The name of the model to use for prediction.
    :type model_name: str

    :return: A tuple containing the predicted class label and its score.
    :rtype: Tuple[str, float]

    :raises ValueError: If the model name or repository ID is invalid.
    :raises RuntimeError: If there's an error during prediction.
    """
    return _open_models_for_repo_id(repo_id).predict(image, model_name)
