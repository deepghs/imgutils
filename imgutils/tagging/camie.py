"""
This module provides functionality for image tagging using the Camie model from Hugging Face Hub.
It includes tools for loading models, processing images, and extracting tags across different categories 
like rating, general tags, characters, artists, and more.
"""
import json
from typing import List, Tuple, Dict, Optional, Union, Literal, Any

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

from .format import remove_underline
from .overlap import drop_overlap_tags
from ..data import ImageTyping, load_image
from ..preprocess import create_pillow_transforms
from ..utils import open_onnx_model, ts_lru_cache, vnames, vreplace

_REPO_ID = 'deepghs/camie_tagger_onnx'
_DEFAULT_MODEL_NAME = 'initial'
_CATEGORY_MAPS = {
    'rating': 9,
    'year': 10,
    'general': 0,
    'artist': 1,
    'copyright': 3,
    'character': 4,
    'meta': 5,
}


@ts_lru_cache()
def _get_camie_model(model_name, is_full: bool):
    """
    Load and cache a Camie ONNX model from the Hugging Face Hub.

    :param model_name: Name of the model to load
    :type model_name: str
    :param is_full: Whether to load the full model or initial-only version
    :type is_full: bool
    :return: Loaded ONNX model
    :rtype: ONNXModel
    """
    return open_onnx_model(hf_hub_download(
        repo_id=_REPO_ID,
        repo_type='model',
        filename=f'{model_name}/model.onnx' if is_full else f'{model_name}/model_initial_only.onnx',
    ))


@ts_lru_cache()
def _get_camie_labels(model_name, no_underline: bool = False) -> Tuple[List[str], Dict[str, List[int]]]:
    """
    Retrieve and process labels for the Camie model.

    :param model_name: Name of the model
    :type model_name: str
    :param no_underline: If True, replace underscores with spaces in tag names
    :type no_underline: bool
    :return: Tuple of (list of tag names, dictionary mapping category names to their indices)
    :rtype: Tuple[List[str], Dict[str, List[int]]]
    """
    path = hf_hub_download(
        repo_id=_REPO_ID,
        repo_type='model',
        filename=f'{model_name}/selected_tags.csv',
    )
    df = pd.read_csv(path)
    name_series = df["name"]
    if no_underline:
        name_series = name_series.map(remove_underline)
    tag_names: List[str] = name_series.tolist()

    indices = {
        cate_name: list(np.where(df["category"] == category)[0])
        for cate_name, category in _CATEGORY_MAPS.items()
    }
    return tag_names, indices


@ts_lru_cache()
def _get_camie_preprocessor(model_name: str):
    """
    Get the image preprocessor for the specified Camie model.

    :param model_name: Name of the model
    :type model_name: str
    :return: Pillow transform pipeline for image preprocessing
    :rtype: callable
    """
    with open(hf_hub_download(
            repo_id=_REPO_ID,
            repo_type='model',
            filename=f'{model_name}/preprocess.json',
    ), 'r') as f:
        p = json.load(f)['stages']
        return create_pillow_transforms(p)


CamieModeTyping = Literal['balanced', 'high_precision', 'high_recall', 'micro_opt', 'macro_opt']


@ts_lru_cache()
def _get_camie_threshold(model_name: str, mode: CamieModeTyping = 'balanced'):
    """
    Get threshold values for different categories based on the specified mode.

    :param model_name: Name of the model
    :type model_name: str
    :param mode: Prediction mode affecting threshold values
    :type mode: CamieModeTyping
    :return: Dictionary of thresholds for each category
    :rtype: Dict[str, float]
    """
    with open(hf_hub_download(
            repo_id=_REPO_ID,
            repo_type='model',
            filename=f'{model_name}/threshold.json',
    ), 'r') as f:
        raw = json.load(f)
        return {
            cate_name: raw[cate_name][mode]['threshold']
            for cate_name in _CATEGORY_MAPS.keys()
        }


def _postprocess_embedding_values(
        pred, logits, embedding,
        model_name: str = _DEFAULT_MODEL_NAME,
        mode: CamieModeTyping = 'balanced',
        thresholds: Optional[Union[float, Dict[str, float]]] = None,
        no_underline: bool = False,
        drop_overlap: bool = False,
):
    """
    Post-process model predictions and embeddings into structured tag results.

    :param pred: Raw prediction array from the model
    :type pred: numpy.ndarray
    :param logits: Logits array from the model
    :type logits: numpy.ndarray
    :param embedding: Embedding array from the model
    :type embedding: numpy.ndarray
    :param model_name: Name of the model used
    :type model_name: str
    :param mode: Prediction mode for threshold selection
    :type mode: CamieModeTyping
    :param thresholds: Custom thresholds for tag selection
    :type thresholds: Optional[Union[float, Dict[str, float]]]
    :param no_underline: Whether to remove underscores from tag names
    :type no_underline: bool
    :param drop_overlap: Whether to remove overlapping tags
    :type drop_overlap: bool
    :return: Dictionary containing processed predictions and embeddings
    :rtype: Dict[str, Any]
    """
    assert len(pred.shape) == len(embedding.shape) == 1, \
        f'Both pred and embeddings shapes should be 1-dim, ' \
        f'but pred: {pred.shape!r}, embedding: {embedding.shape!r} actually found.'
    tag_names, indices = _get_camie_labels(model_name, no_underline)
    labels = list(zip(tag_names, pred.astype(float)))

    default_thresholds = _get_camie_threshold(model_name=model_name, mode=mode)

    rating = {labels[i][0]: labels[i][1].item() for i in indices['rating']}
    tags, d_tags = {}, {}

    for cate_name, index in indices.items():
        if cate_name == 'rating':
            continue

        if thresholds is not None:
            if isinstance(thresholds, dict):
                threshold = thresholds.get(cate_name, default_thresholds[cate_name])
            elif isinstance(thresholds, (int, float)):
                threshold = thresholds
            else:
                raise TypeError(f'Unknown thresholds type for camie tagger - {thresholds!r}.')
        else:
            threshold = default_thresholds[cate_name]

        names = [labels[i] for i in index]
        cate_pred = {x: v.item() for x, v in names if v >= threshold}
        if cate_name == 'general' and drop_overlap:
            cate_pred = drop_overlap_tags(cate_pred)
        tags.update(cate_pred)
        d_tags[cate_name] = cate_pred

    return {
        'rating': rating,
        **d_tags,
        'tag': tags,
        'embedding': embedding.astype(np.float32),
        'logits': logits.astype(np.float32),
        'prediction': pred.astype(np.float32),
    }


def get_camie_tags(
        image: ImageTyping,
        model_name: str = _DEFAULT_MODEL_NAME,
        mode: CamieModeTyping = 'balanced',
        thresholds: Optional[Union[float, Dict[str, float]]] = None,
        no_underline: bool = False,
        drop_overlap: bool = False,
        fmt: Any = ('rating', 'general', 'character'),
):
    """
    Extract tags from an image using the Camie model.

    :param image: Input image (can be path, URL, or image data)
    :type image: ImageTyping
    :param model_name: Name of the Camie model to use
    :type model_name: str
    :param mode: Prediction mode affecting threshold values
    :type mode: CamieModeTyping
    :param thresholds: Custom thresholds for tag selection
    :type thresholds: Optional[Union[float, Dict[str, float]]]
    :param no_underline: Whether to remove underscores from tag names
    :type no_underline: bool
    :param drop_overlap: Whether to remove overlapping tags
    :type drop_overlap: bool
    :param fmt: Format specification for output
    :type fmt: Any
    :return: Dictionary of extracted tags and embeddings
    :rtype: Dict[str, Any]
    """
    names = vnames(fmt)
    need_full = False
    for name in names:
        if '/' in name and name.split('/')[0] == 'initial':
            pass  # is initial
        else:
            need_full = True
            break

    image = load_image(image, force_background='white', mode='RGB')
    input_ = _get_camie_preprocessor(model_name)(image)[np.newaxis, ...]

    if need_full:
        model = _get_camie_model(model_name, is_full=True)
        init_embedding, init_logits, init_pred, refined_embedding, refined_logits, refined_pred = \
            model.run(["initial/embedding", "initial/logits", "initial/output", "embedding", "logits", "output"],
                      {'input': input_})
        init_values = _postprocess_embedding_values(
            pred=init_pred[0],
            logits=init_logits[0],
            embedding=init_embedding[0],
            model_name=model_name,
            mode=mode,
            thresholds=thresholds,
            drop_overlap=drop_overlap,
        )
        refined_values = _postprocess_embedding_values(
            pred=refined_pred[0],
            logits=refined_logits[0],
            embedding=refined_embedding[0],
            model_name=model_name,
            mode=mode,
            thresholds=thresholds,
            no_underline=no_underline,
            drop_overlap=drop_overlap,
        )
        values = {
            **refined_values,
            **{f'initial/{key}': value for key, value in init_values.items()},
            **{f'refined/{key}': value for key, value in refined_values.items()},
        }

    else:
        model = _get_camie_model(model_name, is_full=False)
        init_embedding, init_logits, init_pred = \
            model.run(["embedding", "logits", "output"], {'input': input_})
        init_values = _postprocess_embedding_values(
            pred=init_pred[0],
            logits=init_logits[0],
            embedding=init_embedding[0],
            model_name=model_name,
            mode=mode,
            thresholds=thresholds,
            no_underline=no_underline,
            drop_overlap=drop_overlap,
        )
        values = {
            **{f'initial/{key}': value for key, value in init_values.items()},
        }

    return vreplace(fmt, values)
