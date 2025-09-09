"""
Overview:
    This module provides utilities for image tagging using PixAI taggers, which are specialized models
    for analyzing anime-style images and extracting relevant tags. The module supports loading ONNX
    models from Hugging Face Hub and processing images to generate categorized tags with confidence scores.

    The models are originally developed by the PixAI team and available at 
    `pixai-labs <https://huggingface.co/pixai-labs>`_ on Hugging Face. This module uses ONNX-converted
    versions of these models for efficient inference, available at 
    `deepghs <https://huggingface.co/deepghs>`_ repositories.

    In addition to standard tagging, the models can identify anime character IP (Intellectual Property)
    associations. For example, if a character like "misaka_mikoto" is detected, the system can map
    this to the "toaru_kagaku_no_railgun" (A Certain Scientific Railgun) IP. All IP names follow
    Danbooru-style tag conventions for consistency with existing anime tagging systems.

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

        >>> # Get IP information for detected characters
        >>> ips = get_pixai_tags('path/to/image.jpg', fmt='ips')
        >>> print("Detected IPs:", ips)
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

    This function downloads and loads the CSV file containing tag names, categories,
    and IP (Intellectual Property) associations for the specified model. The DataFrame
    contains columns for tag names, categories, and other metadata including character
    IP mappings when available.

    :param model_name: Name of the model
    :type model_name: str

    :return: Tuple containing (DataFrame with tag information, dictionary mapping character tags to their IPs)
    :rtype: Tuple[pd.DataFrame, Dict[str, List[str]]]
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
    :param model_name: Name or repository ID of the PixAI tagger model to use
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
        - ``ips_mapping``: a dictionary mapping detected character tags to their associated
          IP (Intellectual Property) names in Danbooru-style format
        - ``ips_count``: a dictionary containing IP names and their occurrence counts based
          on detected characters
        - ``ips``: a list of IP names sorted by occurrence count (descending) and name
          (ascending), representing the most likely anime/game series in the image

        Default category thresholds are used if not specified. These vary by model and category
        but typically range from 0.35 to 0.5.

        You can extract embedding of the given image with the following code

        >>> from imgutils.tagging import get_pixai_tags
        >>>
        >>> embedding = get_pixai_tags('skadi.jpg', fmt='embedding')
        >>> embedding.shape
        (1024, )

        This embedding is valuable for constructing indices that enable rapid querying of images based on
        visual features within large-scale datasets.

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
        >>>
        >>> # Get IP information for character identification
        >>> ips_mapping = get_pixai_tags('image.jpg', fmt='ips_mapping')
        >>> print("Character to IP mapping:", ips_mapping)
        >>> # Example output: {'misaka_mikoto': ['toaru_kagaku_no_railgun'], 'hu_tao_(genshin_impact)': ['genshin_impact']}
        >>>
        >>> # Get most likely anime/game series
        >>> top_ips = get_pixai_tags('image.jpg', fmt='ips')
        >>> print("Most likely series:", top_ips)
        >>> # Example output: ['genshin_impact', 'toaru_kagaku_no_railgun']

        Here are some images for example

        .. image:: tagging_demo.plot.py.svg
           :align: center

        >>> general, character = get_pixai_tags('skadi.jpg')
        >>> general
        {'patreon_username': 0.9988852739334106, 'baseball_bat': 0.9977256059646606, 'holding_baseball_bat': 0.9858889579772949, 'navel': 0.9830228090286255, 'crop_top': 0.9666315317153931, 'sportswear': 0.9664723873138428, '1girl': 0.9572311639785767, 'long_hair': 0.9550737738609314, 'outdoors': 0.9501817226409912, 'solo': 0.9466996788978577, 'day': 0.9394471049308777, 'breasts': 0.938787579536438, 'web_address': 0.9387772679328918, 'stomach': 0.935083270072937, 'red_eyes': 0.9326196908950806, 'shorts': 0.9305683374404907, 'motion_blur': 0.9278550148010254, 'playing_sports': 0.9263769388198853, 'blue_sky': 0.9213213920593262, 'midriff': 0.9191423654556274, 'large_breasts': 0.9174768924713135, 'artist_name': 0.9089528322219849, 'sky': 0.9054281711578369, 'baseball': 0.904181957244873, 'gloves': 0.9033604860305786, 'thighs': 0.893738865852356, 'black_shorts': 0.8926981687545776, 'volleyball': 0.8198539614677429, 'very_long_hair': 0.7967187166213989, 'short_shorts': 0.7873305082321167, 'black_gloves': 0.7765249013900757, 'white_hair': 0.770541787147522, 'baseball_mitt': 0.7684446573257446, 'thigh_gap': 0.73811936378479, 'sweat': 0.7263807654380798, 'cowboy_shot': 0.7235408425331116, 'short_sleeves': 0.7062878012657166, 'parted_lips': 0.7025120258331299, 'patreon_logo': 0.6970672607421875, 'cloud': 0.6967148780822754, 'looking_at_viewer': 0.6898926496505737, 'holding': 0.6879030466079712, 'swinging': 0.6736525893211365, 'ass_visible_through_thighs': 0.6636734008789062, 'elbow_pads': 0.6630151867866516, 'shirt': 0.6250661611557007, 'hair_between_eyes': 0.6075361967086792, 'standing': 0.5285079479217529, 'black_shirt': 0.5173394680023193, 'linea_alba': 0.513701319694519, 'baseball_uniform': 0.48175835609436035, 'crop_top_overhang': 0.4682246744632721, 'ball': 0.43616628646850586, 'blurry': 0.4201475977897644, 'baseball_stadium': 0.41493287682533264, 'grey_hair': 0.39384859800338745, 'watermark': 0.3919041156768799, 'black_sports_bra': 0.3877854645252228, 'fanbox_username': 0.3790855407714844, 'narrow_waist': 0.36392998695373535}
        >>> character
        {'skadi_(arknights)': 0.8926791548728943}
        >>>
        >>> general, character = get_pixai_tags('hutao.jpg')
        >>> general
        {'bag': 0.9833353757858276, 'backpack': 0.9766197204589844, 'flower-shaped_pupils': 0.962916910648346, 'tongue_out': 0.960152804851532, 'tongue': 0.9526823163032532, 'ghost': 0.9514724016189575, 'plaid_skirt': 0.9499615430831909, '1girl': 0.9378864765167236, 'skirt': 0.9353114366531372, 'bag_charm': 0.9314961433410645, 'symbol-shaped_pupils': 0.9252510070800781, 'charm_(object)': 0.9249529242515564, 'twintails': 0.9239017367362976, 'flower': 0.9175764322280884, 'outdoors': 0.9175151586532593, 'hair_ornament': 0.9161680936813354, 'plaid_clothes': 0.9144806861877441, 'long_hair': 0.8768749833106995, 'pleated_skirt': 0.8597153425216675, 'school_uniform': 0.8573414087295532, 'looking_at_viewer': 0.8392735719680786, ':p': 0.8193913698196411, 'hair_between_eyes': 0.8070638179779053, 'hair_flower': 0.8054562211036682, 'nail_polish': 0.8011300563812256, 'building': 0.7961824536323547, 'jacket': 0.7647742629051208, 'brown_hair': 0.7541910409927368, 'solo': 0.7539198398590088, 'long_sleeves': 0.7471930980682373, 'ahoge': 0.7171378135681152, 'hair_ribbon': 0.6994943618774414, 'red_eyes': 0.6819245219230652, 'bowtie': 0.6639955043792725, 'sidelocks': 0.6275356411933899, 'bush': 0.6164096593856812, 'gate': 0.612525224685669, 'smile': 0.6077383160591125, 'shirt': 0.6042965650558472, 'contemporary': 0.5968752503395081, 'brick_floor': 0.5933602452278137, 'cardigan': 0.5810519456863403, 'gradient_hair': 0.5570307970046997, 'diagonal-striped_bowtie': 0.5565160512924194, 'alternate_costume': 0.5535630583763123, 'school_bag': 0.5535626411437988, 'black_hair': 0.5434530973434448, 'ribbon': 0.5332301259040833, 'hairclip': 0.523446261882782, 'day': 0.5164296627044678, 'street': 0.49987316131591797, 'bow': 0.4941294193267822, 'plum_blossoms': 0.4940766990184784, 'collared_shirt': 0.49013280868530273, 'standing': 0.4820355772972107, 'blue_cardigan': 0.47836002707481384, 'cowboy_shot': 0.4782080054283142, 'pocket': 0.477585107088089, 'pavement': 0.4712265729904175, 'multicolored_hair': 0.4610708951950073, 'blue_jacket': 0.45271334052085876, 'blush': 0.45005902647972107, 'sleeves_past_wrists': 0.440649151802063, 'black_nails': 0.4402858316898346, 'black_bag': 0.4206739366054535, 'miniskirt': 0.4187837243080139, 'red_bow': 0.414681613445282, 'very_long_hair': 0.4129619002342224, 'diagonal-striped_clothes': 0.4112803339958191, 'blazer': 0.40750616788864136, 'striped_bowtie': 0.40123170614242554, 'sunlight': 0.4008329212665558, 'grey_skirt': 0.3930213749408722, 'road': 0.3819067180156708, 'black_ribbon': 0.3776353895664215, 'thighs': 0.3722286820411682, 'hug': 0.37215015292167664, 'brick_wall': 0.3717171549797058, 'white_shirt': 0.3694952428340912, 'open_clothes': 0.36442798376083374, 'open_jacket': 0.3525886535644531, ':d': 0.3343055844306946, 'multicolored_nails': 0.32190075516700745, 'red_bowtie': 0.3157669007778168, 'star-shaped_pupils': 0.309164822101593, 'open_mouth': 0.30890953540802, 'beads': 0.3084579110145569, 'stone_stairs': 0.30559882521629333, 'randoseru': 0.30517613887786865}
        >>> character
        {'hu_tao_(genshin_impact)': 0.9997367858886719, 'boo_tao_(genshin_impact)': 0.999537467956543}
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
    if 'ips' in df_tags.columns:
        ips_mapping, ips_counts = {}, defaultdict(lambda: 0)
        for tag, _ in tags.items():
            if tag in d_ips:
                ips_mapping[tag] = d_ips[tag]
                for ip_name in d_ips[tag]:
                    ips_counts[ip_name] += 1
        values['ips_mapping'] = ips_mapping
        values['ips_count'] = dict(ips_counts)
        values['ips'] = [x for x, _ in sorted(ips_counts.items(), key=lambda x: (-x[1], x[0]))]
    return vreplace(fmt, values)
