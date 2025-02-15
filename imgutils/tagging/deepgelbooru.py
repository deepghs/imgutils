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
    return open_onnx_model(hf_hub_download(
        repo_id=_REPO_ID,
        repo_type='model',
        filename='model.onnx',
    ))


@ts_lru_cache()
def _open_preprocessor():
    with open(hf_hub_download(
            repo_id=_REPO_ID,
            repo_type='model',
            filename='preprocessor.json',
    ), 'r') as f:
        return create_pillow_transforms(json.load(f)['stages'])


@ts_lru_cache()
def _open_tags():
    df_tags = pd.read_csv(hf_hub_download(
        repo_id=_REPO_ID,
        repo_type='model',
        filename='tags.csv',
    ))
    return {item['tag_id']: item for item in df_tags.to_dict('records')}


def _image_preprocess(image: Image.Image):
    return _open_preprocessor()(image).transpose((1, 2, 0))[None, ...].astype(np.float32)


_PREFIX_LENGTH = len('rating:')


def get_deepgelbooru_tags(image: ImageTyping,
                          general_threshold: float = 0.3, character_threshold: float = 0.3,
                          drop_overlap: bool = False, fmt=('rating', 'general', 'character')):
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
            d_rating[tag_info['name'][_PREFIX_LENGTH:]] = score
        else:
            assert False, 'Should not reach this line.'  # pragma: no cover

    if drop_overlap:
        d_general = drop_overlap_tags(d_general)

    return vreplace(fmt, {
        'general': d_general,
        'character': d_characters,
        'rating': d_rating,
        'prediction': prediction,
    })
