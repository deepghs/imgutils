"""
Overview:
    Tagging utils based on deepgelbooru.

    Inspired from `LagPixelLOL/deepgelbooru <https://github.com/LagPixelLOL/deepgelbooru>`_
    trained by `@LagPixelLOL <https://github.com/LagPixelLOL>`_.

    ONNX model is hosted on `deepghs/deepgelbooru_onnx <https://huggingface.co/deepghs/deepgelbooru_onnx>`_.
"""
import json

import numpy as np
import pandas as pd
from PIL import Image
from huggingface_hub import hf_hub_download

from .overlap import drop_overlap_tags
from ..data import ImageTyping, load_image
from ..preprocess import create_pillow_transforms
from ..utils import ts_lru_cache, open_onnx_model, vreplace

_REPO_ID = 'deepghs/deepgelbooru_onnx'


@ts_lru_cache()
def _open_model():
    """
    Open and cache the DeepGelbooru ONNX model.

    :return: The loaded ONNX model session
    :rtype: onnxruntime.InferenceSession
    """
    return open_onnx_model(hf_hub_download(
        repo_id=_REPO_ID,
        repo_type='model',
        filename='model.onnx',
    ))


@ts_lru_cache()
def _open_preprocessor():
    """
    Load and cache the image preprocessing configuration.

    :return: A callable transform pipeline for image preprocessing
    :rtype: callable
    """
    with open(hf_hub_download(
            repo_id=_REPO_ID,
            repo_type='model',
            filename='preprocessor.json',
    ), 'r') as f:
        return create_pillow_transforms(json.load(f)['stages'])


@ts_lru_cache()
def _open_tags():
    """
    Load and cache the tag definitions from CSV.

    :return: Dictionary mapping tag IDs to tag information
    :rtype: dict
    """
    df_tags = pd.read_csv(hf_hub_download(
        repo_id=_REPO_ID,
        repo_type='model',
        filename='tags.csv',
    ))
    return {item['tag_id']: item for item in df_tags.to_dict('records')}


def _image_preprocess(image: Image.Image):
    """
    Preprocess an image for model inference.

    :param image: PIL Image to preprocess
    :type image: PIL.Image.Image
    :return: Preprocessed image array
    :rtype: numpy.ndarray
    """
    return _open_preprocessor()(image).transpose((1, 2, 0))[None, ...].astype(np.float32)


def get_deepgelbooru_tags(image: ImageTyping,
                          general_threshold: float = 0.3, character_threshold: float = 0.3,
                          drop_overlap: bool = False, fmt=('rating', 'general', 'character')):
    """
    Extract tags from an image using the DeepGelbooru model.

    :param image: Input image (can be PIL Image, path, or bytes)
    :type image: ImageTyping
    :param general_threshold: Confidence threshold for general tags
    :type general_threshold: float
    :param character_threshold: Confidence threshold for character tags
    :type character_threshold: float
    :param drop_overlap: Whether to remove overlapping tags
    :type drop_overlap: bool
    :param fmt: Format of the output, specifying which tag categories to include

    :return: Dictionary containing predicted tags by category

    .. note::
        The fmt argument can include the following keys:

        - ``rating``: a dict containing ratings and their confidences
        - ``general``: a dict containing general tags and their confidences
        - ``character``: a dict containing character tags and their confidences
        - ``tag``: a dict containing all tags (including general and character, not including rating) and their confidences
        - ``prediction``: a 1-dim prediction result of image

    Example:
        Here are some images for example

        .. image:: tagging_demo.plot.py.svg
           :align: center

        >>> from imgutils.tagging import get_deepgelbooru_tags
        >>>
        >>> rating, features, chars = get_deepgelbooru_tags('skadi.jpg')
        >>> rating
        {'rating:safe': 0.9986732006072998, 'rating:questionable': 0.0013858973979949951, 'rating:explicit': 4.315376281738281e-05}
        >>> features
        {'1girl': 0.9972434639930725, 'baseball': 0.5982598662376404, 'baseball_bat': 0.6429562568664551, 'bike_shorts': 0.36296138167381287, 'black_gloves': 0.8308937549591064, 'black_shirt': 0.7388008832931519, 'blue_sky': 0.6039759516716003, 'blush': 0.30909663438796997, 'breasts': 0.9694308042526245, 'cloud': 0.6422968506813049, 'cowboy_shot': 0.5898381471633911, 'crop_top': 0.8145260810852051, 'day': 0.652222216129303, 'dolphin_shorts': 0.466494083404541, 'gloves': 0.7183809280395508, 'hair_between_eyes': 0.6753682494163513, 'holding': 0.7302790880203247, 'holding_baseball_bat': 0.6649775505065918, 'large_breasts': 0.8446108102798462, 'long_hair': 0.98187655210495, 'looking_at_viewer': 0.8140730857849121, 'midriff': 0.6360533833503723, 'navel': 0.9635934829711914, 'no_hat': 0.33370012044906616, 'no_headwear': 0.44239571690559387, 'outdoors': 0.7891374826431274, 'parted_lips': 0.6471294164657593, 'red_eyes': 0.9958090782165527, 'shirt': 0.8736815452575684, 'short_sleeves': 0.872096061706543, 'shorts': 0.5640895366668701, 'silver_hair': 0.5049663186073303, 'sky': 0.8832778930664062, 'solo': 0.9687467813491821, 'sports_bra': 0.3659853935241699, 'sportswear': 0.9309735298156738, 'standing': 0.49939480423927307, 'stomach': 0.446407288312912, 'sweat': 0.809670090675354, 'thighs': 0.6560589075088501, 'very_long_hair': 0.47956040501594543}
        >>> chars
        {'skadi_(arknights)': 0.9832853674888611}
        >>>
        >>> rating, features, chars = get_deepgelbooru_tags('hutao.jpg')
        >>> rating
        {'rating:safe': 0.9994162321090698, 'rating:questionable': 0.0008397102355957031, 'rating:explicit': 0.00035390257835388184}
        >>> features
        {'1girl': 0.9926226139068604, ':p': 0.899387001991272, 'ahoge': 0.34215790033340454, 'backpack': 0.5701972246170044, 'bag': 0.9512913227081299, 'bag_charm': 0.6664570569992065, 'blush': 0.5614628791809082, 'bow': 0.33615976572036743, 'breasts': 0.5770801305770874, 'brown_hair': 0.987317681312561, 'buttons': 0.37286585569381714, 'cardigan': 0.36409223079681396, 'charm_(object)': 0.7329680919647217, 'collared_shirt': 0.5924292206764221, 'cowboy_shot': 0.4344901144504547, 'flower': 0.7465001344680786, 'hair_between_eyes': 0.5225946307182312, 'hair_flower': 0.6976451873779297, 'hair_ornament': 0.9265321493148804, 'hair_ribbon': 0.34527891874313354, 'jacket': 0.6675043106079102, 'long_hair': 0.9096828699111938, 'long_sleeves': 0.41341525316238403, 'looking_at_viewer': 0.8418735265731812, 'miniskirt': 0.3675632178783417, 'nail_polish': 0.5284417867660522, 'open_clothes': 0.30296844244003296, 'outdoors': 0.48789578676223755, 'plaid': 0.36596980690956116, 'plaid_skirt': 0.7759367227554321, 'pleated_skirt': 0.6535028219223022, 'red_eyes': 0.8975257873535156, 'ribbon': 0.36911237239837646, 'school_bag': 0.4171145558357239, 'school_uniform': 0.3942635953426361, 'shirt': 0.6772940754890442, 'skirt': 0.9397937655448914, 'sleeves_past_wrists': 0.5207280516624451, 'smile': 0.4673041105270386, 'solo': 0.9118321537971497, 'tongue': 0.9967410564422607, 'tongue_out': 0.9970728158950806, 'twintails': 0.8419480323791504, 'very_long_hair': 0.6489560604095459, 'white_shirt': 0.6217572689056396}
        >>> chars
        {}
    """
    input_ = _image_preprocess(load_image(image, mode='RGB'))
    session = _open_model()
    prediction, = session.run(['prediction'], {'input': input_})
    prediction = prediction[0]

    d_tags = _open_tags()
    d_general, d_characters, d_rating = {}, {}, {}
    for idx, score in enumerate(prediction.tolist()):
        tag_info = d_tags[idx]
        category = tag_info['category']
        if category == 0:
            if score >= general_threshold:
                d_general[tag_info['name']] = score
        elif category == 4:
            if score >= character_threshold:
                d_characters[tag_info['name']] = score
        elif category == 9:
            d_rating[tag_info['name']] = score
        else:
            assert False, 'Should not reach this line.'  # pragma: no cover

    if drop_overlap:
        d_general = drop_overlap_tags(d_general)

    return vreplace(fmt, {
        'general': d_general,
        'character': d_characters,
        'rating': d_rating,
        'tag': {**d_general, **d_characters},
        'prediction': prediction,
    })
