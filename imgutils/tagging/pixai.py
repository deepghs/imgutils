import json
from typing import List, Tuple, Any

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

from imgutils.data import load_image, ImageTyping
from imgutils.preprocess import create_pillow_transforms
from imgutils.tagging.format import remove_underline
from imgutils.tagging.overlap import drop_overlap_tags
from imgutils.utils import open_onnx_model, vreplace, ts_lru_cache

EXP_REPO = 'onopix/pixai-tagger-onnx'
MODEL_FILENAME = "model.onnx"
LABEL_FILENAME = "selected_tags.csv"

_DEFAULT_MODEL_NAME = 'tagger_v_2_2_7'


@ts_lru_cache()
def _get_pixai_model(model_name):
    """
    Load an ONNX model from the Hugging Face Hub.

    :param model_name: The name of the model to load.
    :type model_name: str
    :return: The loaded ONNX model.
    :rtype: ONNXModel
    """
    return open_onnx_model(hf_hub_download(
        repo_id=EXP_REPO,
        filename=f'{model_name}/model.onnx',
    ))


@ts_lru_cache()
def _get_pixai_labels(model_name, no_underline: bool = False) -> Tuple[List[str], List[int], List[int]]:
    """
    Get labels for the pixai model.

    :param model_name: The name of the model.
    :type model_name: str
    :param no_underline: If True, replaces underscores in tag names with spaces.
    :type no_underline: bool
    :return: A tuple containing the list of tag names, and lists of indexes for rating, general, and character categories.
    :rtype: Tuple[List[str], List[int], List[int]]
    """
    df = pd.read_csv(hf_hub_download(
        repo_id=EXP_REPO,
        filename=f'{model_name}/selected_tags.csv',
    ))
    name_series = df["name"]
    if no_underline:
        name_series = name_series.map(remove_underline)
    tag_names = name_series.tolist()

    general_indexes = list(np.where(df["category"] == 0)[0])
    character_indexes = list(np.where(df["category"] == 4)[0])
    return tag_names, general_indexes, character_indexes


@ts_lru_cache()
def _get_pixai_weights(model_name):
    """
    Load the weights for a pixai model.

    :param model_name: The name of the model.
    :type model_name: str
    :return: The loaded weights.
    :rtype: numpy.ndarray
    """
    return np.load(hf_hub_download(
        repo_id=EXP_REPO,
        filename=f'{model_name}/matrix.npz',
    ))


@ts_lru_cache()
def _open_preprocess_transforms(model_name: str):
    with open(hf_hub_download(
            repo_id=EXP_REPO,
            filename=f'{model_name}/preprocess.json',
    )) as f:
        return create_pillow_transforms(json.load(f)['stages'])


def _prepare_image_for_tagging(image: ImageTyping, model_name: str):
    """
    Prepare an image for tagging by resizing and padding it.

    :param image: The input image.
    :type image: ImageTyping
    :param model_name: Name of the model.
    :type model_name: str
    :return: The prepared image as a numpy array.
    :rtype: numpy.ndarray
    """
    image = load_image(image, force_background='white', mode='RGB')
    image_array = _open_preprocess_transforms(model_name)(image)
    return np.expand_dims(image_array, axis=0)


def _postprocess_embedding(
        pred, embedding, logit,
        model_name: str = _DEFAULT_MODEL_NAME,
        general_threshold: float = 0.15,
        character_threshold: float = 0.7,
        no_underline: bool = False,
        drop_overlap: bool = False,
        fmt: Any = ('general', 'character'),
):
    """
    Post-process the embedding and prediction results.

    :param pred: The prediction array.
    :type pred: numpy.ndarray
    :param embedding: The embedding array.
    :type embedding: numpy.ndarray
    :param logit: The logit array.
    :type logit: numpy.ndarray
    :param model_name: The name of the model used.
    :type model_name: str
    :param general_threshold: Threshold for general tags.
    :type general_threshold: float
    :param character_threshold: Threshold for character tags.
    :type character_threshold: float
    :param no_underline: Whether to remove underscores from tag names.
    :type no_underline: bool
    :param drop_overlap: Whether to drop overlapping tags.
    :type drop_overlap: bool
    :param fmt: The format of the output.
    :type fmt: Any
    :return: The post-processed results.
    """
    assert len(pred.shape) == len(embedding.shape) == 1, \
        f'Both pred and embeddings shapes should be 1-dim, ' \
        f'but pred: {pred.shape!r}, embedding: {embedding.shape!r} actually found.'
    tag_names, general_indexes, character_indexes = _get_pixai_labels(model_name, no_underline)
    labels = list(zip(tag_names, pred.astype(float)))

    general_names = [labels[i] for i in general_indexes]
    general_res = {x: v.item() for x, v in general_names if v > general_threshold}
    if drop_overlap:
        general_res = drop_overlap_tags(general_res)

    character_names = [labels[i] for i in character_indexes]
    character_res = {x: v.item() for x, v in character_names if v > character_threshold}

    return vreplace(
        fmt,
        {
            'general': general_res,
            'character': character_res,
            'tag': {**general_res, **character_res},
            'embedding': embedding.astype(np.float32),
            'prediction': pred.astype(np.float32),
            'logit': logit.astype(np.float32),
        }
    )


def get_pixai_tags(
        image: ImageTyping,
        model_name: str = _DEFAULT_MODEL_NAME,
        general_threshold: float = 0.15,
        character_threshold: float = 0.7,
        no_underline: bool = False,
        drop_overlap: bool = False,
        fmt: Any = ('general', 'character'),
):
    """
    Get tags for an image using pixai taggers.

    :param image: The input image.
    :type image: ImageTyping
    :param model_name: The name of the model to use.
    :type model_name: str
    :param general_threshold: The threshold for general tags.
    :type general_threshold: float
    :param character_threshold: The threshold for character tags.
    :type character_threshold: float
    :param no_underline: If True, replaces underscores in tag names with spaces.
    :type no_underline: bool
    :param drop_overlap: If True, drops overlapping tags.
    :type drop_overlap: bool
    :param fmt: Return format, default is ``('general', 'character')``.
        ``embedding`` is also supported for feature extraction.
    :type fmt: Any
    :return: Prediction result based on the provided fmt.

    .. note::
        The fmt argument can include the following keys:

        - ``rating``: a dict containing ratings and their confidences
        - ``general``: a dict containing general tags and their confidences
        - ``character``: a dict containing character tags and their confidences
        - ``tag``: a dict containing all tags (including general and character, not including rating) and their confidences
        - ``embedding``: a 1-dim embedding of image, recommended for index building after L2 normalization
        - ``logit``: a 1-dim logit of image, before softmax.
        - ``prediction``: a 1-dim prediction result of image

        You can extract embedding of the given image with the follwing code

        >>> from imgutils.tagging import get_pixai_tags
        >>>
        >>> embedding = get_pixai_tags('pixai/1.jpg', fmt='embedding')
        >>> embedding.shape
        (1024, )

        This embedding is valuable for constructing indices that enable rapid querying of images based on
        visual features within large-scale datasets.
    """

    model = _get_pixai_model(model_name)
    _, _, target_size, _ = model.get_inputs()[0].shape
    input_ = _prepare_image_for_tagging(image, model_name=model_name)
    preds, logits, embeddings = model.run(['output', 'logits', 'embedding'], {'input': input_})

    return _postprocess_embedding(
        pred=preds[0],
        embedding=embeddings[0],
        logit=logits[0],
        model_name=model_name,
        general_threshold=general_threshold,
        character_threshold=character_threshold,
        no_underline=no_underline,
        drop_overlap=drop_overlap,
        fmt=fmt,
    )
