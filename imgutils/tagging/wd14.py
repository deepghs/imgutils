"""
Overview:
    Tagging utils based on wd14 v2, inspired by
    `SmilingWolf/wd-v1-4-tags <https://huggingface.co/spaces/SmilingWolf/wd-v1-4-tags>`_ .
"""
from functools import lru_cache
from typing import List, Tuple

import cv2
import huggingface_hub
import numpy as np
import pandas as pd

from ..data import load_image, ImageTyping
from ..utils import open_onnx_model


def make_square(img, target_size):
    old_size = img.shape[:2]
    desired_size = max(old_size)
    desired_size = max(desired_size, target_size)

    delta_w = desired_size - old_size[1]
    delta_h = desired_size - old_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [255, 255, 255]
    # noinspection PyUnresolvedReferences
    new_im = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return new_im


def smart_resize(img, size):
    # Assumes the image has already gone through make_square
    if img.shape[0] > size:
        # noinspection PyUnresolvedReferences
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    elif img.shape[0] < size:
        # noinspection PyUnresolvedReferences
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    return img


SWIN_MODEL_REPO = "SmilingWolf/wd-v1-4-swinv2-tagger-v2"
CONV_MODEL_REPO = "SmilingWolf/wd-v1-4-convnext-tagger-v2"
CONV2_MODEL_REPO = "SmilingWolf/wd-v1-4-convnextv2-tagger-v2"
VIT_MODEL_REPO = "SmilingWolf/wd-v1-4-vit-tagger-v2"
MODEL_FILENAME = "model.onnx"
LABEL_FILENAME = "selected_tags.csv"

MODEL_NAMES = {
    "SwinV2": SWIN_MODEL_REPO,
    "ConvNext": CONV_MODEL_REPO,
    "ConvNextV2": CONV2_MODEL_REPO,
    "ViT": VIT_MODEL_REPO
}


def _load_wd14_model(model_repo: str, model_filename: str):
    return open_onnx_model(huggingface_hub.hf_hub_download(model_repo, model_filename))


@lru_cache()
def _get_wd14_model(model_name):
    return _load_wd14_model(MODEL_NAMES[model_name], MODEL_FILENAME)


@lru_cache()
def _get_wd14_labels() -> Tuple[List[str], List[int], List[int], List[int]]:
    path = huggingface_hub.hf_hub_download(CONV2_MODEL_REPO, LABEL_FILENAME)
    df = pd.read_csv(path)

    tag_names = df["name"].tolist()
    rating_indexes = list(np.where(df["category"] == 9)[0])
    general_indexes = list(np.where(df["category"] == 0)[0])
    character_indexes = list(np.where(df["category"] == 4)[0])
    return tag_names, rating_indexes, general_indexes, character_indexes


def get_wd14_tags(image: ImageTyping, model_name: str = "ConvNextV2",
                  general_threshold: float = 0.35, character_threshold: float = 0.85):
    """
    Overview:
        Tagging image by wd14 v2 model. Similar to
        `SmilingWolf/wd-v1-4-tags <https://huggingface.co/spaces/SmilingWolf/wd-v1-4-tags>`_ .

    :param image: Image to tagging.
    :param model_name: Name of the mode, should be one of the \
        ``SwinV2``, ``ConvNext``, ``ConvNextV2`` or ``ViT``, default is ``ConvNextV2``.
    :param general_threshold: Threshold for default tags, default is ``0.35``.
    :param character_threshold: Threshold for character tags, default is ``0.85``.
    :return: Tagging results for levels, features and characters.

    Example:
        Here are some images for example

        .. image:: tagging_demo.plot.py.svg
           :align: center

        >>> import os
        >>> from imgutils.tagging import get_wd14_tags
        >>>
        >>> rating, features, chars = get_wd14_tags('skadi.jpg')
        >>> rating
        {'general': 0.0011444687843322754, 'sensitive': 0.8876402974128723, 'questionable': 0.106781005859375, 'explicit': 0.000277101993560791}
        >>> features
        {'1girl': 0.997527003288269, 'solo': 0.9797663688659668, 'long_hair': 0.9905703663825989, 'breasts': 0.9761719703674316, 'looking_at_viewer': 0.8981098532676697, 'bangs': 0.8810765743255615, 'large_breasts': 0.9498510360717773, 'shirt': 0.8377365469932556, 'red_eyes': 0.945058286190033, 'gloves': 0.9457170367240906, 'navel': 0.969594419002533, 'holding': 0.7881088852882385, 'hair_between_eyes': 0.7687551379203796, 'very_long_hair': 0.9301245212554932, 'standing': 0.6703325510025024, 'white_hair': 0.5292627811431885, 'short_sleeves': 0.8677047491073608, 'grey_hair': 0.5859264731407166, 'thighs': 0.9536856412887573, 'cowboy_shot': 0.8056888580322266, 'sweat': 0.8394746780395508, 'outdoors': 0.9473626613616943, 'parted_lips': 0.8986269235610962, 'sky': 0.9385137557983398, 'shorts': 0.8408567905426025, 'alternate_costume': 0.4245271384716034, 'day': 0.931140661239624, 'black_gloves': 0.8830795884132385, 'midriff': 0.7279844284057617, 'artist_name': 0.5333830714225769, 'cloud': 0.64717698097229, 'stomach': 0.9516432285308838, 'blue_sky': 0.9655293226242065, 'crop_top': 0.9485014081001282, 'black_shirt': 0.7366660833358765, 'short_shorts': 0.7161656618118286, 'ass_visible_through_thighs': 0.5858667492866516, 'black_shorts': 0.6186309456825256, 'thigh_gap': 0.41193312406539917, 'no_headwear': 0.467605859041214, 'low-tied_long_hair': 0.36282333731651306, 'sportswear': 0.3756745457649231, 'motion_blur': 0.5091936588287354, 'baseball_bat': 0.951993465423584, 'baseball': 0.5634750723838806, 'holding_baseball_bat': 0.8232709169387817}
        >>> chars
        {'skadi_(arknights)': 0.9869340658187866}
        >>>
        >>> rating, features, chars = get_wd14_tags('hutao.jpg')
        >>> rating
        {'general': 0.49491602182388306, 'sensitive': 0.5193622708320618, 'questionable': 0.003406703472137451, 'explicit': 0.0007208287715911865}
        >>> features
        {'1girl': 0.9798132181167603, 'solo': 0.8046203851699829, 'long_hair': 0.7596215009689331, 'looking_at_viewer': 0.7620116472244263, 'blush': 0.46084529161453247, 'smile': 0.48454540967941284, 'bangs': 0.5152207016944885, 'skirt': 0.8023070096969604, 'brown_hair': 0.8653596639633179, 'hair_ornament': 0.7201820611953735, 'red_eyes': 0.7816740870475769, 'long_sleeves': 0.697688639163971, 'twintails': 0.8974947333335876, 'school_uniform': 0.7491052746772766, 'jacket': 0.5015512704849243, 'flower': 0.6401398181915283, 'ahoge': 0.43420469760894775, 'pleated_skirt': 0.4528769850730896, 'outdoors': 0.5730487704277039, 'tongue': 0.6739872694015503, 'hair_flower': 0.5545973181724548, 'tongue_out': 0.6946243047714233, 'bag': 0.5487751364707947, 'symbol-shaped_pupils': 0.7439308166503906, 'blazer': 0.4186026453971863, 'backpack': 0.47378358244895935, ':p': 0.4690653085708618, 'ghost': 0.7565015554428101}
        >>> chars
        {'hu_tao_(genshin_impact)': 0.9262397289276123, 'boo_tao_(genshin_impact)': 0.942080020904541}
    """
    model = _get_wd14_model(model_name)
    _, height, width, _ = model.get_inputs()[0].shape

    # Load image, PIL RGB to OpenCV BGR
    image = load_image(image, mode='RGB', force_background='white')
    image = np.asarray(image)[:, :, ::-1]

    image = make_square(image, height)
    image = smart_resize(image, height)
    image = image.astype(np.float32)
    image = np.expand_dims(image, 0)

    input_name = model.get_inputs()[0].name
    label_name = model.get_outputs()[0].name
    probs = model.run([label_name], {input_name: image})[0]

    tag_names, rating_indexes, general_indexes, character_indexes = _get_wd14_labels()
    labels = list(zip(tag_names, probs[0].astype(float).tolist()))

    # First 4 labels are actually ratings: pick one with argmax
    ratings_names = [labels[i] for i in rating_indexes]
    rating = dict(ratings_names)

    # Then we have general tags: pick anywhere prediction confidence > threshold
    general_names = [labels[i] for i in general_indexes]
    general_res = [x for x in general_names if x[1] > general_threshold]
    general_res = dict(general_res)

    # Everything else is characters: pick anywhere prediction confidence > threshold
    character_names = [labels[i] for i in character_indexes]
    character_res = [x for x in character_names if x[1] > character_threshold]
    character_res = dict(character_res)

    return rating, general_res, character_res
