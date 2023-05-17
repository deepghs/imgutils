from functools import lru_cache
from typing import Union, List

import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from sklearn.cluster import DBSCAN
from tqdm.auto import tqdm

from ..data import MultiImagesTyping, load_images, ImageTyping
from ..utils import open_onnx_model

__all__ = [
    'get_ccip_features',
    'get_ccip_similarity',
    'batch_ccip_similarity',
    'ccip_clustering',
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
        f'deepghs/imgutils-models',
        f'ccip/{model_name}_feat.onnx',
    ))


@lru_cache()
def _open_metric_model(model_name, safe: bool = False):
    return open_onnx_model(hf_hub_download(
        f'deepghs/imgutils-models',
        f'ccip/{model_name}_{"safe_" if safe else ""}metrics.onnx',
    ))


_VALID_MODEL_NAMES = [
    'ccip-caformer-5_fp32',
    'ccip-caformer-4_fp32',
    'ccip-caformer-2_fp32',
]
_DEFAULT_MODEL_NAMES = _VALID_MODEL_NAMES[0]


def get_ccip_features(images: MultiImagesTyping, size: int = 384, model_name: str = _DEFAULT_MODEL_NAMES):
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
                feats.append(get_ccip_features(load_images([item]), size, model_name)[0])

        return np.stack(feats)
    else:
        raise TypeError(f'Unknown feature batch type - {x!r}.')


_FeatureOrImage = Union[ImageTyping, np.ndarray]


def get_ccip_similarity(x: _FeatureOrImage, y: _FeatureOrImage, safe: bool = False,
                        size: int = 384, model_name: str = _DEFAULT_MODEL_NAMES) -> float:
    return batch_ccip_similarity([x, y], safe, size, model_name)[0, 1].item()


def batch_ccip_similarity(images: Union[np.ndarray, List[_FeatureOrImage]], safe: bool = False,
                          size: int = 384, model_name: str = _DEFAULT_MODEL_NAMES):
    input_ = _preprocess_feats(images, size, model_name).astype(np.float32)
    output, = _open_metric_model(model_name, safe=safe).run(['output'], {'input': input_})
    return output


def ccip_clustering(images: MultiImagesTyping, threshold: float = 0.6, safe: bool = True,
                    size: int = 384, model_name: str = _DEFAULT_MODEL_NAMES):
    images = load_images(images, mode='RGB')
    features = []
    for image in tqdm(images, desc='Feature Extract'):
        features.append(get_ccip_features([image], size, model_name)[0])

    if not features:
        return []
    feats = np.stack(features)
    differences = 1 - batch_ccip_similarity(feats, safe, size, model_name)

    def _metric(x, y):
        return differences[int(x), int(y)]

    samples = np.array(range(len(images))).reshape(-1, 1)
    clustering = DBSCAN(eps=1 - threshold, min_samples=2, metric=_metric).fit(samples)
    return clustering.labels_.tolist()
