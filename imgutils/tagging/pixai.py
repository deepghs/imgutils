import json
from typing import Union, Dict, Any

import pandas as pd
from hbutils.design import SingletonMark
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

from imgutils.data import ImageTyping, load_image
from imgutils.preprocess import create_pillow_transforms
from imgutils.utils import open_onnx_model, ts_lru_cache, vreplace

FMT_UNSET = SingletonMark('FMT_UNSET')


def _get_repo_id(model_name: str) -> str:
    if '/' in model_name:
        return model_name
    else:
        return f'deepghs/pixai-tagger-{model_name}-onnx'


@ts_lru_cache()
def _open_onnx_model(model_name: str):
    """
    Load the ONNX model from Hugging Face Hub.

    :return: The loaded ONNX model
    :rtype: object
    """
    return open_onnx_model(hf_hub_download(
        repo_id=_get_repo_id(model_name),
        repo_type='model',
        filename='model.onnx',
    ))


@ts_lru_cache()
def _open_tags(model_name: str) -> pd.DataFrame:
    return pd.read_csv(hf_hub_download(
        repo_id=_get_repo_id(model_name),
        repo_type='model',
        filename='selected_tags.csv',
    ))


@ts_lru_cache()
def _open_preprocess(model_name: str):
    with open(hf_hub_download(
            repo_id=_get_repo_id(model_name),
            repo_type='model',
            filename='preprocess.json'
    ), 'r') as f:
        data_ = json.load(f)
        return create_pillow_transforms(data_['stages'])


@ts_lru_cache()
def _open_default_category_thresholds(model_name: str) -> Union[Dict[int, float], Dict[int, str]]:
    """
    Load default category thresholds from the Hugging Face Hub.

    :return: Dictionary mapping category IDs to threshold values
    :rtype: dict
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
    Make a raw prediction with the model.

    :param image: The input image
    :type image: ImageTyping
    :param preprocessor: Which preprocessor to use ('test' or 'val')
    :type preprocessor: Literal['test', 'val']

    :return: Dictionary of model outputs
    :rtype: dict
    :raises ValueError: If an unknown preprocessor is specified
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
    Make a prediction and format the results.

    This method processes an image through the model and applies thresholds to determine
    which tags to include in the results. The output format can be customized using the fmt parameter.

    :param image: The input image
    :type image: ImageTyping
    :param preprocessor: Which preprocessor to use ('test' or 'val')
    :type preprocessor: Literal['test', 'val']
    :param thresholds: Threshold values for tag confidence. Can be a single float applied to all categories
                      or a dictionary mapping category IDs or names to threshold values
    :type thresholds: Union[float, Dict[Any, float]]
    :param use_tag_thresholds: Whether to use tag-level thresholds if available
    :type use_tag_thresholds: bool
    :param fmt: Output format specification. Can be a tuple of category names to include,
               or FMT_UNSET to use all categories
    :type fmt: Any

    :return: Formatted prediction results according to the fmt parameter
    :rtype: Any

    .. note::
        The fmt argument can include the following keys:

        - Category names: dicts containing category-specific tags and their confidences
        - ``tag``: a dict containing all tags across categories and their confidences
        - ``embedding``: a 1-dim embedding of image, recommended for index building after L2 normalization
        - ``logits``: a 1-dim logits result of image
        - ``prediction``: a 1-dim prediction result of image

        You can extract specific category predictions or all tags based on your needs.

    For more details see documentation of :func:`multilabel_timm_predict`.
    """
    df_tags = _open_tags(model_name=model_name)
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
    return vreplace(fmt, values)
