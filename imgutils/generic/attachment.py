"""
This module provides functionality for handling attachments in machine learning models,
particularly those hosted on Hugging Face's model hub. It includes tools for loading,
managing and making predictions with ONNX models for classification, tagging and regression tasks.

The module provides a caching mechanism for model loading and thread-safe operations
for concurrent access to models and their metadata.

An example of attachment models is `deepghs/eattach_monochrome_experiments <https://huggingface.co/deepghs/eattach_monochrome_experiments>`_.

.. note::
    If you want to train a custom attachment model for taggers,
    take a look at our framework `deepghs/emb_attachments <https://github.com/deepghs/emb_attachments>`_.
"""

import json
import os
from threading import Lock
from typing import Optional, Any, Tuple

import numpy as np
from huggingface_hub import hf_hub_download

from ..utils import open_onnx_model, vreplace, ts_lru_cache


class Attachment:
    """
    A class to manage machine learning model attachments from Hugging Face.

    This class handles model loading, caching, and prediction for various types of problems
    including classification, tagging, and regression.

    :param repo_id: The Hugging Face repository ID
    :type repo_id: str
    :param model_name: Name of the model
    :type model_name: str
    :param hf_token: Optional Hugging Face authentication token
    :type hf_token: Optional[str]
    """

    def __init__(self, repo_id: str, model_name: str, hf_token: Optional[str] = None):
        """
        Initialize the Attachment instance with repository and model information.
        """
        self.repo_id = repo_id
        self.model_name = model_name
        self._meta_value = None
        self._model = None

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
    def _meta(self):
        """
        Load and cache model metadata from the Hugging Face repository.

        :return: Model metadata as a dictionary
        :rtype: dict
        """
        with self._model_lock:
            if self._meta_value is None:
                with open(hf_hub_download(
                        repo_id=self.repo_id,
                        repo_type='model',
                        filename=f'{self.model_name}/meta.json',
                        token=self._get_hf_token(),
                ), 'r') as f:
                    self._meta_value = json.load(f)

        return self._meta_value

    @property
    def encoder_model(self) -> str:
        """
        Get the encoder model name from metadata.

        :return: Name of the encoder model
        :rtype: str
        """
        return self._meta['encoder_model']

    def _open_model(self):
        """
        Load and cache the ONNX model from Hugging Face.

        :return: Loaded ONNX model
        :rtype: object
        """
        with self._model_lock:
            if self._model is None:
                self._model = open_onnx_model(hf_hub_download(
                    repo_id=self.repo_id,
                    repo_type='model',
                    filename=f'{self.model_name}/model.onnx',
                    token=self._get_hf_token(),
                ))

        return self._model

    def _predict_raw(self, embedding: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make raw predictions using the model.

        :param embedding: Input embedding array
        :type embedding: np.ndarray
        :return: Tuple of logits and predictions
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        model = self._open_model()
        logits, prediction = model.run(['logits', 'prediction'], {'input': embedding})
        return logits, prediction

    def _predict_classification(self, embedding: np.ndarray, fmt: Any = 'top'):
        """
        Make classification predictions.

        :param embedding: Input embedding array
        :type embedding: np.ndarray
        :param fmt: Format specification for output
        :type fmt: Any
        :return: List of formatted prediction results
        :rtype: list
        """
        labels = np.array(self._meta['problem']['labels'])
        logits, prediction = self._predict_raw(embedding)
        retval = []
        for logit, pred in zip(logits, prediction):
            scores = dict(zip(labels, pred.tolist()))
            maxidx = np.argmax(pred)
            top_label, top_score = labels[maxidx].item(), pred[maxidx].item()
            top = top_label, top_score
            retval.append(vreplace(fmt, {
                'scores': scores,
                'top': top,
                'top_label': top_label,
                'top_score': top_score,
                'logit': logit,
                'prediction': pred,
            }))

        return retval

    def _predict_tagging(self, embedding: np.ndarray, threshold: float = 0.3, fmt: Any = 'tags'):
        """
        Make tagging predictions.

        :param embedding: Input embedding array
        :type embedding: np.ndarray
        :param threshold: Confidence threshold for tag selection
        :type threshold: float
        :param fmt: Format specification for output
        :type fmt: Any
        :return: List of formatted prediction results
        :rtype: list
        """
        tags = np.array(self._meta['problem']['tags'])
        logits, prediction = self._predict_raw(embedding)
        retval = []
        for logit, pred in zip(logits, prediction):
            selection = pred >= threshold
            pvalues, ptags = pred[selection], tags[selection]
            result = dict(zip(ptags.tolist(), pvalues.tolist()))
            retval.append(vreplace(fmt, {
                'tags': result,
                'logit': logit,
                'prediction': pred,
            }))

        return retval

    def _predict_regression(self, embedding: np.ndarray, fmt: Any = 'full'):
        """
        Make regression predictions.

        :param embedding: Input embedding array
        :type embedding: np.ndarray
        :param fmt: Format specification for output
        :type fmt: Any
        :return: List of formatted prediction results
        :rtype: list
        """
        field_names = [name for name, _, _ in self._meta['problem']['fields']]
        logits, prediction = self._predict_raw(embedding)
        retval = []
        for logit, pred in zip(logits, prediction):
            result = dict(zip(field_names, pred.tolist()))
            retval.append(vreplace(fmt, {
                'full': result,
                'logit': logit,
                'prediction': pred,
                **{f'field/{key}': value for key, value in result.items()},
            }))

        return retval

    def predict(self, embedding: np.ndarray, **kwargs):
        """
        Make predictions based on the problem type (classification, tagging, or regression).

        :param embedding: Input embedding array
        :type embedding: np.ndarray
        :param kwargs: Additional arguments passed to specific prediction methods
        :return: Prediction results in specified format
        :raises ValueError: If embedding shape is invalid or problem type is unknown
        """
        embedding = embedding.astype(np.float32)
        if len(embedding.shape) == 1:
            single = True
            embedding = embedding[np.newaxis, ...]
        elif len(embedding.shape) == 2:
            single = False
        else:
            raise ValueError(f'Unexpected embedding shape - {embedding!r}.')

        problem_type = self._meta['problem']['type']
        if problem_type == 'classification':
            result = self._predict_classification(embedding, **kwargs)
        elif problem_type == 'tagging':
            result = self._predict_tagging(embedding, **kwargs)
        elif problem_type == 'regression':
            result = self._predict_regression(embedding, **kwargs)
        else:
            raise ValueError(f'Unknown problem type - {problem_type!r}.')

        if single:
            result = result[0]
        return result


@ts_lru_cache()
def open_attachment(repo_id: str, model_name: str, hf_token: Optional[str] = None) -> 'Attachment':
    """
    Create and cache an Attachment instance.

    This function creates a new Attachment instance or returns a cached one
    if it was previously created with the same parameters.

    :param repo_id: The Hugging Face repository ID
    :type repo_id: str
    :param model_name: Name of the model
    :type model_name: str
    :param hf_token: Optional Hugging Face authentication token
    :type hf_token: Optional[str]
    :return: An Attachment instance
    :rtype: Attachment
    """
    return Attachment(
        repo_id=repo_id,
        model_name=model_name,
        hf_token=hf_token,
    )
