import json
from functools import lru_cache
from typing import Union, List, Optional, Tuple

import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from sklearn.cluster import DBSCAN, OPTICS
from tqdm.auto import tqdm

try:
    from typing import Literal
except (ModuleNotFoundError, ImportError):
    from typing_extensions import Literal

from ..data import MultiImagesTyping, load_images, ImageTyping
from ..utils import open_onnx_model

__all__ = [
    'ccip_extract_feature',
    'ccip_batch_extract_features',

    'ccip_default_threshold',
    'ccip_difference',
    'ccip_same',
    'ccip_batch_differences',
    'ccip_batch_same',

    'ccip_default_clustering_params',
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


_VALID_MODEL_NAMES = [
    'ccip-caformer-24-randaug-pruned',
    'ccip-caformer-6-randaug-pruned_fp32',
    'ccip-caformer-5_fp32',
]
_DEFAULT_MODEL_NAMES = 'ccip-caformer-24-randaug-pruned'


def ccip_extract_feature(image: ImageTyping, size: int = 384, model_name: str = _DEFAULT_MODEL_NAMES):
    return ccip_batch_extract_features([image], size, model_name)[0]


def ccip_batch_extract_features(images: MultiImagesTyping, size: int = 384, model_name: str = _DEFAULT_MODEL_NAMES):
    images = load_images(images, mode='RGB')
    data = np.stack([_preprocess_image(item, size=size) for item in images]).astype(np.float32)
    output, = _open_feat_model(model_name).run(['output'], {'input': data})
    return output


_FeatureOrImage = Union[ImageTyping, np.ndarray]


def _p_feature(x: _FeatureOrImage, size: int = 384, model_name: str = _DEFAULT_MODEL_NAMES):
    if isinstance(x, np.ndarray):  # if feature
        return x
    else:  # is image or path
        return ccip_extract_feature(x, size, model_name)


def ccip_default_threshold(model_name: str = _DEFAULT_MODEL_NAMES) -> float:
    return _open_metrics(model_name)['threshold']


def ccip_difference(x: _FeatureOrImage, y: _FeatureOrImage,
                    size: int = 384, model_name: str = _DEFAULT_MODEL_NAMES) -> float:
    return ccip_batch_differences([x, y], size, model_name)[0, 1].item()


def ccip_same(x: _FeatureOrImage, y: _FeatureOrImage, threshold: Optional[float] = None,
              size: int = 384, model_name: str = _DEFAULT_MODEL_NAMES) -> float:
    diff = ccip_difference(x, y, size, model_name)
    threshold = threshold if threshold is not None else ccip_default_threshold(model_name)
    return diff <= threshold


def ccip_batch_differences(images: List[_FeatureOrImage],
                           size: int = 384, model_name: str = _DEFAULT_MODEL_NAMES) -> np.ndarray:
    input_ = np.stack([_p_feature(img, size, model_name) for img in images]).astype(np.float32)
    output, = _open_metric_model(model_name).run(['output'], {'input': input_})
    return output


def ccip_batch_same(images: List[_FeatureOrImage], threshold: Optional[float] = None,
                    size: int = 384, model_name: str = _DEFAULT_MODEL_NAMES) -> np.ndarray:
    batch_diff = ccip_batch_differences(images, size, model_name)
    threshold = threshold if threshold is not None else ccip_default_threshold(model_name)
    return batch_diff <= threshold


CCIPClusterModeTyping = Literal['dbscane', 'dbscan_2', 'dbscan_free', 'optics']


def ccip_default_clustering_params(model_name: str = _DEFAULT_MODEL_NAMES,
                                   mode: CCIPClusterModeTyping = 'dbscan') -> Tuple[float, int]:
    if mode == 'dbscan':
        return ccip_default_threshold(model_name), 2
    else:
        _info = _open_cluster_metrics(model_name)[mode]
        return _info['eps'], _info['min_samples']


def ccip_clustering(images: List[_FeatureOrImage], mode: CCIPClusterModeTyping = 'dbscan',
                    eps: Optional[float] = None, min_samples: Optional[int] = None,
                    size: int = 384, model_name: str = _DEFAULT_MODEL_NAMES) -> np.ndarray:
    _default_eps, _default_min_samples = ccip_default_clustering_params(model_name, mode)
    eps = eps or _default_eps
    min_samples = min_samples or _default_min_samples

    images = [_p_feature(img, size, model_name) for img in tqdm(images, desc='Extract features')]
    batch_diff = ccip_batch_differences(images, size, model_name)

    def _metric(x, y):
        return batch_diff[int(x), int(y)].item()

    samples = np.arange(len(images)).reshape(-1, 1)
    if 'dbscan' in mode:
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric=_metric).fit(samples)
    elif mode == 'optics':
        clustering = OPTICS(max_eps=eps, min_samples=min_samples, metric=_metric).fit(samples)
    else:
        raise ValueError(f'Unknown mode for CCIP clustering - {mode!r}.')

    return clustering.labels_.tolist()
