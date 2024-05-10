"""
Overview:
    Generic tools for classification models.
"""
import json
import os
from functools import lru_cache
from typing import Tuple, Optional, List, Dict

import numpy as np
from PIL import Image
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
    Encode an image into a numpy array.

    :param image: The input image.
    :type image: Image.Image
    :param size: The size to resize the image to, defaults to (384, 384).
    :type size: Tuple[int, int], optional
    :param normalize: The mean and standard deviation for normalization, defaults to (0.5, 0.5).
    :type normalize: Optional[Tuple[float, float]], optional

    :return: The encoded image as a numpy array.
    :rtype: np.ndarray
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
    Class for managing classification models.

    This class provides methods for loading classification models, predicting scores, and predictions.

    Methods:
        predict_score: Predicts the scores for each class.
        predict: Predicts the class with the highest score.
        clear: Clears the loaded models and labels.

    Attributes:
        None
    """

    def __init__(self, repo_id: str):
        """
        Initialize the ClassifyModel instance.

        :param repo_id: The repository ID containing the models.
        :type repo_id: str
        """
        self.repo_id = repo_id
        self._model_names = None
        self._models = {}
        self._labels = {}

    @classmethod
    def _get_hf_token(cls):
        """
        Get the Hugging Face token from the environment variable.

        :return: The Hugging Face token.
        :rtype: str
        """
        return os.environ.get('HF_TOKEN')

    @property
    def model_names(self) -> List[str]:
        """
        Get the model names available in the repository.

        :return: The list of model names.
        :rtype: List[str]
        """
        if self._model_names is None:
            hf_fs = HfFileSystem(token=self._get_hf_token())
            self._model_names = [
                os.path.dirname(os.path.relpath(item, self.repo_id)) for item in
                hf_fs.glob(f'{self.repo_id}/*/model.onnx')
            ]

        return self._model_names

    def _check_model_name(self, model_name: str):
        """
        Check if the model name is valid.

        :param model_name: The name of the model.
        :type model_name: str

        :raises ValueError: If the model name is invalid.
        """
        if model_name not in self.model_names:
            raise ValueError(f'Unknown model {model_name!r} in model repository {self.repo_id!r}, '
                             f'models {self.model_names!r} are available.')

    def _open_model(self, model_name: str):
        """
        Open the specified model.

        :param model_name: The name of the model.
        :type model_name: str

        :return: The opened model.
        :rtype: Any
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
        Open the labels file for the specified model.

        :param model_name: The name of the model.
        :type model_name: str

        :return: The list of labels.
        :rtype: List[str]
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

        :param image: The input image.
        :type image: ImageTyping
        :param model_name: The name of the model.
        :type model_name: str

        :return: The raw prediction.
        :rtype: np.ndarray
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
        Predict the scores for each class.

        :param image: The input image.
        :type image: ImageTyping
        :param model_name: The name of the model.
        :type model_name: str

        :return: The dictionary containing class scores.
        :rtype: Dict[str, float]
        """
        output = self._raw_predict(image, model_name)
        values = dict(zip(self._open_label(model_name), map(lambda x: x.item(), output[0])))
        return values

    def predict(self, image: ImageTyping, model_name: str) -> Tuple[str, float]:
        """
        Predict the class with the highest score.

        :param image: The input image.
        :type image: ImageTyping
        :param model_name: The name of the model.
        :type model_name: str

        :return: The predicted class and its score.
        :rtype: Tuple[str, float]
        """
        output = self._raw_predict(image, model_name)[0]
        max_id = np.argmax(output)
        return self._open_label(model_name)[max_id], output[max_id].item()

    def clear(self):
        """
        Clear the loaded models and labels.
        """
        self._models.clear()
        self._labels.clear()


@lru_cache()
def _open_models_for_repo_id(repo_id: str) -> ClassifyModel:
    """
    Open classification models for the specified repository ID.

    :param repo_id: The repository ID containing the models.
    :type repo_id: str

    :return: The ClassifyModel instance for the repository.
    :rtype: ClassifyModel
    """
    return ClassifyModel(repo_id)


def classify_predict_score(image: ImageTyping, repo_id: str, model_name: str) -> Dict[str, float]:
    """
    Predict the scores for each class using the specified model.

    :param image: The input image.
    :type image: ImageTyping
    :param repo_id: The repository ID containing the models.
    :type repo_id: str
    :param model_name: The name of the model.
    :type model_name: str

    :return: The dictionary containing class scores.
    :rtype: Dict[str, float]
    """
    return _open_models_for_repo_id(repo_id).predict_score(image, model_name)


def classify_predict(image: ImageTyping, repo_id: str, model_name: str) -> Tuple[str, float]:
    """
    Predict the class with the highest score using the specified model.

    :param image: The input image.
    :type image: ImageTyping
    :param repo_id: The repository ID containing the models.
    :type repo_id: str
    :param model_name: The name of the model.
    :type model_name: str

    :return: The predicted class and its score.
    :rtype: Tuple[str, float]
    """
    return _open_models_for_repo_id(repo_id).predict(image, model_name)
