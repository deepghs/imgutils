import json
from functools import lru_cache
from typing import Union, List

import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

from ..data import MultiImagesTyping, load_images, ImageTyping
from ..utils import open_onnx_model

__all__ = [
    'get_ccip_feature',
    'batch_ccip_features',
    'get_ccip_difference',
    'batch_ccip_differences',
]


def _normalize(data, mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)):
    mean, std = np.asarray(mean), np.asarray(std)
    return (data - mean[:, None, None]) / std[:, None, None]


def _preprocess_image(image: Image.Image, size: int = 384):
    image = image.resize((size, size), resample=Image.BILINEAR)
    # noinspection PyTypeChecker
    data = np.array(image).transpose(2, 0, 1).astype(np.float32) / 255.0
    data = _normalize(data)

    return data


@lru_cache()
def _open_feat_model(model_name):
    return open_onnx_model(hf_hub_download(
        f'deepghs/ccip_onnx',
        f'{model_name}/model_feat.onnx',
    ))


@lru_cache()
def _open_metric_model(model_name):
    return open_onnx_model(hf_hub_download(
        f'deepghs/ccip_onnx',
        f'{model_name}/model_metrics.onnx',
    ))


@lru_cache()
def _open_metrics(model_name):
    with open(hf_hub_download(f'deepghs/ccip_onnx', f'{model_name}/metrics.json'), 'r') as f:
        return json.load(f)


@lru_cache()
def _open_cluster_metrics(model_name):
    with open(hf_hub_download(f'deepghs/ccip_onnx', f'{model_name}/cluster.json'), 'r') as f:
        return json.load(f)


_DEFAULT_MODEL_NAMES = 'ccip-caformer-24-randaug-pruned'


def get_ccip_feature(image: ImageTyping, size: int = 384, model_name: str = _DEFAULT_MODEL_NAMES):
    return batch_ccip_features([image], size, model_name)[0]


def batch_ccip_features(images: MultiImagesTyping, size: int = 384, model_name: str = _DEFAULT_MODEL_NAMES):
    images = load_images(images, mode='RGB')
    data = np.stack([_preprocess_image(item, size=size) for item in images]).astype(np.float32)
    output, = _open_feat_model(model_name).run(['output'], {'input': data})
    return output


def _preprocess_feats(x, size: int = 384, model_name: str = _DEFAULT_MODEL_NAMES):
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, (list, tuple)):
        feats = []
        for item in x:
            if isinstance(item, np.ndarray):
                feats.append(item)
            else:
                feats.append(batch_ccip_features(load_images([item]), size, model_name)[0])

        return np.stack(feats)
    else:
        raise TypeError(f'Unknown feature batch type - {x!r}.')


_FeatureOrImage = Union[ImageTyping, np.ndarray]


def get_ccip_difference(x: _FeatureOrImage, y: _FeatureOrImage,
                        size: int = 384, model_name: str = _DEFAULT_MODEL_NAMES) -> float:
    return batch_ccip_differences([x, y], size, model_name)[0, 1].item()


def batch_ccip_differences(images: Union[np.ndarray, List[_FeatureOrImage]],
                           size: int = 384, model_name: str = _DEFAULT_MODEL_NAMES):
    input_ = _preprocess_feats(images, size, model_name).astype(np.float32)
    output, = _open_metric_model(model_name).run(['output'], {'input': input_})
    return output
