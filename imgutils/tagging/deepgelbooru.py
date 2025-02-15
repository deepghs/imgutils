import json

import pandas as pd
from huggingface_hub import hf_hub_download

from ..preprocess import create_pillow_transforms
from ..utils import ts_lru_cache, open_onnx_model

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

