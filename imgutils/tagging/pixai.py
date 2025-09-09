"""
Overview:
    This module provides utilities for image tagging using PixAI taggers, which are specialized models
    for analyzing anime-style images and extracting relevant tags. The module supports loading ONNX
    models from Hugging Face Hub and processing images to generate categorized tags with confidence scores.

    The models are originally developed by the PixAI team and available at 
    `pixai-labs <https://huggingface.co/pixai-labs>`_ on Hugging Face. This module uses ONNX-converted
    versions of these models for efficient inference, available at 
    `deepghs <https://huggingface.co/deepghs>`_ repositories.

    Example::
        >>> from imgutils.tagging.pixai import get_pixai_tags
        >>> # Get tags with default thresholds
        >>> result = get_pixai_tags('path/to/anime_image.jpg', model_name='v0.9')
        >>> general_tags, character_tags = result
        >>> print("General tags:", general_tags)
        >>> print("Character tags:", character_tags)

        >>> # Get all tags in a single dictionary
        >>> all_tags = get_pixai_tags('path/to/image.jpg', fmt='tag')
        >>> print("All tags:", all_tags)
"""

import json
from collections import defaultdict
from typing import Union, Dict, Any, Tuple, List

import pandas as pd
from hbutils.design import SingletonMark
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

from imgutils.data import ImageTyping, load_image
from imgutils.preprocess import create_pillow_transforms
from imgutils.utils import open_onnx_model, ts_lru_cache, vreplace

FMT_UNSET = SingletonMark('FMT_UNSET')


def _get_repo_id(model_name: str) -> str:
    """
    Get the repository ID for the specified model name.

    :param model_name: Name of the model (e.g., 'v0.9') or full repository path
    :type model_name: str

    :return: Full repository ID for Hugging Face Hub
    :rtype: str

    Example::
        >>> _get_repo_id('v0.9')
        'deepghs/pixai-tagger-v0.9-onnx'
        >>> _get_repo_id('custom/model-repo')
        'custom/model-repo'
    """
    if '/' in model_name:
        return model_name
    else:
        return f'deepghs/pixai-tagger-{model_name}-onnx'


@ts_lru_cache()
def _open_onnx_model(model_name: str):
    """
    Load the ONNX model from Hugging Face Hub with caching.

    This function downloads and loads the ONNX model file for the specified PixAI tagger.
    Results are cached to avoid repeated downloads and model loading.

    :param model_name: Name of the model to load
    :type model_name: str

    :return: The loaded ONNX model session
    :rtype: onnxruntime.InferenceSession
    """
    return open_onnx_model(hf_hub_download(
        repo_id=_get_repo_id(model_name),
        repo_type='model',
        filename='model.onnx',
    ))


@ts_lru_cache()
def _open_tags(model_name: str) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Load the tag metadata from Hugging Face Hub with caching.

    This function downloads and loads the CSV file containing tag names and categories
    for the specified model. The DataFrame contains columns for tag names, categories,
    and other metadata.

    :param model_name: Name of the model
    :type model_name: str

    :return: DataFrame containing tag information with columns like 'name', 'category'
    :rtype: pd.DataFrame
    """
    df_tags = pd.read_csv(hf_hub_download(
        repo_id=_get_repo_id(model_name),
        repo_type='model',
        filename='selected_tags.csv',
    ))
    d_ips = {}
    if 'ips' in df_tags:
        df_tags['ips'] = df_tags['ips'].map(json.loads)
        for name, ips in zip(df_tags['name'], df_tags['ips']):
            if ips:
                d_ips[name] = ips
    return df_tags, d_ips


@ts_lru_cache()
def _open_preprocess(model_name: str):
    """
    Load the preprocessing pipeline configuration from Hugging Face Hub with caching.

    This function downloads the preprocessing configuration and creates a PIL transforms
    pipeline for image preprocessing before model inference.

    :param model_name: Name of the model
    :type model_name: str

    :return: Preprocessing transform pipeline
    """
    with open(hf_hub_download(
            repo_id=_get_repo_id(model_name),
            repo_type='model',
            filename='preprocess.json'
    ), 'r') as f:
        data_ = json.load(f)
        return create_pillow_transforms(data_['stages'])


@ts_lru_cache()
def _open_default_category_thresholds(model_name: str) -> Tuple[Dict[int, float], Dict[int, str]]:
    """
    Load default category thresholds and names from the Hugging Face Hub with caching.

    This function attempts to load predefined threshold values for each category from
    a CSV file. If the file doesn't exist, empty dictionaries are returned.

    :param model_name: Name of the model
    :type model_name: str

    :return: Tuple containing (category_thresholds, category_names) dictionaries
    :rtype: tuple[Dict[int, float], Dict[int, str]]

    Example::
        >>> thresholds, names = _open_default_category_thresholds('v0.9')
        >>> print(thresholds)  # {0: 0.35, 1: 0.4, ...}
        >>> print(names)      # {0: 'general', 1: 'character', ...}
    """
    _default_category_thresholds: Dict[int, float] = {}
    _category_names: Dict[int, str] = {}
    try:
        df_category_thresholds = pd.read_csv(hf_hub_download(
            repo_id=_get_repo_id(model_name),
            repo_type='model',
            filename='thresholds.csv'
        ), keep_default_na=False)
    except (EntryNotFoundError,):
        pass
    else:
        for item in df_category_thresholds.to_dict('records'):
            if item['category'] not in _default_category_thresholds:
                _default_category_thresholds[item['category']] = item['threshold']
            _category_names[item['category']] = item['name']

    return _default_category_thresholds, _category_names


def _raw_predict(image: ImageTyping, model_name: str):
    """
    Make a raw prediction with the PixAI tagger model.

    This function preprocesses the input image and runs inference using the specified
    ONNX model. It returns the raw model outputs without any post-processing or
    threshold application.

    :param image: The input image to analyze
    :type image: ImageTyping
    :param model_name: Name of the model to use for prediction
    :type model_name: str

    :return: Dictionary containing raw model outputs with keys like 'prediction', 'embedding', 'logits'
    :rtype: dict

    Example::
        >>> raw_output = _raw_predict('anime_image.jpg', 'v0.9')
        >>> print(raw_output.keys())  # dict_keys(['prediction', 'embedding', 'logits'])
    """
    image = load_image(image, force_background='white', mode='RGB')
    model = _open_onnx_model(model_name=model_name)
    trans = _open_preprocess(model_name=model_name)
    input_ = trans(image)[None, ...]
    output_names = [output.name for output in model.get_outputs()]
    output_values = model.run(output_names, {'input': input_})
    return {name: value[0] for name, value in zip(output_names, output_values)}


def get_pixai_tags(image: ImageTyping, model_name: str = 'v0.9',
                   thresholds: Union[float, Dict[Any, float]] = None, fmt=FMT_UNSET):
    """
    Extract tags from an image using PixAI tagger models.

    This function processes an image through a PixAI tagger model and applies confidence
    thresholds to determine which tags to include in the results. The output format can
    be customized to return specific categories or all tags together.

    :param image: The input image to analyze (file path, PIL Image, numpy array, etc.)
    :type image: ImageTyping
    :param model_name: Name or path of the PixAI tagger model to use
    :type model_name: str
    :param thresholds: Confidence threshold values. Can be a single float applied to all 
                      categories, or a dictionary mapping category IDs/names to specific thresholds
    :type thresholds: Union[float, Dict[Any, float]], optional
    :param fmt: Output format specification. If FMT_UNSET, returns all available categories.
               Can be a tuple of category names to include in output
    :type fmt: Any

    :return: Formatted prediction results. Default returns tuple of (general_tags, character_tags, ...)
            based on available categories. Can return custom format based on fmt parameter
    :rtype: Any

    .. note::
        The fmt parameter can include the following keys:

        - Category names (e.g., 'general', 'character'): dictionaries containing category-specific 
          tags and their confidence scores
        - ``tag``: a dictionary containing all tags across categories and their confidences
        - ``embedding``: a 1-dimensional embedding vector of the image, recommended for similarity 
          search after L2 normalization
        - ``logits``: raw 1-dimensional logits output from the model
        - ``prediction``: 1-dimensional prediction probabilities from the model

        Default category thresholds are used if not specified. These vary by model and category
        but typically range from 0.35 to 0.5.

    Example::
        >>> from imgutils.tagging.pixai import get_pixai_tags
        >>> 
        >>> # Get tags with default format (all categories)
        >>> general_tags, character_tags = get_pixai_tags('anime_image.jpg', model_name='v0.9')
        >>> print("General tags:", general_tags)
        >>> print("Character tags:", character_tags)
        >>> 
        >>> # Get all tags in a single dictionary
        >>> all_tags = get_pixai_tags('image.jpg', fmt='tag')
        >>> print("All tags:", all_tags)
        >>> 
        >>> # Use custom thresholds
        >>> result = get_pixai_tags('image.jpg', thresholds={'general': 0.3, 'character': 0.5})
        >>> 
        >>> # Get embedding for similarity search
        >>> embedding = get_pixai_tags('image.jpg', fmt='embedding')
        >>> # Normalize for cosine similarity
        >>> import numpy as np
        >>> normalized_embedding = embedding / np.linalg.norm(embedding)
    """
    df_tags, d_ips = _open_tags(model_name=model_name)
    values = _raw_predict(image, model_name=model_name)
    prediction = values['prediction']
    tags = {}

    default_category_thresholds, category_names = _open_default_category_thresholds(model_name=model_name)
    if fmt is FMT_UNSET:
        fmt = tuple(category_names[category] for category in sorted(set(df_tags['category'].tolist())))

    for category in sorted(set(df_tags['category'].tolist())):
        mask = df_tags['category'] == category
        tag_names = df_tags['name'][mask]
        category_pred = prediction[mask]

        if isinstance(thresholds, float):
            category_threshold = thresholds
        elif isinstance(thresholds, dict) and \
                (category in thresholds or category_names[category] in thresholds):
            if category in thresholds:
                category_threshold = thresholds[category]
            elif category_names[category] in thresholds:
                category_threshold = thresholds[category_names[category]]
            else:
                assert False, 'Should not reach this line'  # pragma: no cover
        else:
            if category in default_category_thresholds:
                category_threshold = default_category_thresholds[category]
            else:
                category_threshold = 0.4

        mask = category_pred >= category_threshold
        tag_names = tag_names[mask].tolist()
        category_pred = category_pred[mask].tolist()
        cate_tags = dict(sorted(zip(tag_names, category_pred), key=lambda x: (-x[1], x[0])))
        values[category_names[category]] = cate_tags
        tags.update(cate_tags)

    values['tag'] = tags
    ip_mapping, ip_counts = {}, defaultdict(lambda: 0)
    if 'ips' in df_tags.columns:
        for tag, _ in tags.items():
            if tag in d_ips[tag]:
                ip_mapping[tag] = d_ips[tag]
                for ip_name in d_ips[tag]:
                    ip_counts[ip_name] += 1
    values['ips_mapping'] = ip_mapping
    values['ips'] = [x for x, _ in sorted(ip_counts.items(), key=lambda x: (-x[1], x[0]))]
    return vreplace(fmt, values)
