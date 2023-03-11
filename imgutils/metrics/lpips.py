from functools import lru_cache
from typing import Tuple, Union, List

import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from sklearn.cluster import DBSCAN
from tqdm.auto import tqdm

from imgutils.data import rgb_encode, MultiImagesTyping, load_images, ImageTyping, load_image
from imgutils.utils import open_onnx_model

__all__ = [
    'lpips_extract_feature',
    'lpips_difference',
    'lpips_clustering',
]


def _image_resize(image: Image.Image, size=400):
    return image.resize((size, size), resample=Image.BILINEAR)


def _image_encode(image: Image.Image):
    return rgb_encode(_image_resize(image))


@lru_cache()
def _lpips_feature_model():
    return open_onnx_model(hf_hub_download(
        'deepghs/imgutils-models',
        'lpips/lpips_feature.onnx',
    ))


def lpips_extract_feature(image: MultiImagesTyping) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    images = load_images(image)
    _encoded = np.stack([_image_encode(image) for image in images])
    features = _lpips_feature_model().run(["feat_0", "feat_1", "feat_2", "feat_3", "feat_4"], {'input': _encoded})
    return tuple(features)


@lru_cache()
def _lpips_diff_model():
    return open_onnx_model(hf_hub_download(
        'deepghs/imgutils-models',
        'lpips/lpips_diff.onnx',
    ))


_FEAT1_NAMES = ["feat_x_0", "feat_x_1", "feat_x_2", "feat_x_3", "feat_x_4"]
_FEAT2_NAMES = ["feat_y_0", "feat_y_1", "feat_y_2", "feat_y_3", "feat_y_4"]


def _batch_lpips_difference(feats1: Tuple[np.ndarray, ...], feats2: Tuple[np.ndarray, ...]) -> np.ndarray:
    output, = _lpips_diff_model().run(
        ["output"],
        {
            **{name: value for name, value in zip(_FEAT1_NAMES, feats1)},
            **{name: value for name, value in zip(_FEAT2_NAMES, feats2)},
        }
    )
    return output


AutoFeatTyping = Union[ImageTyping, Tuple[np.ndarray, ...]]


def _auto_feat(img: AutoFeatTyping):
    if isinstance(img, (tuple, list)):
        return img
    else:
        return lpips_extract_feature(load_image(img))


def lpips_difference(img1: AutoFeatTyping, img2: AutoFeatTyping) -> float:
    img1 = _auto_feat(img1)
    img2 = _auto_feat(img2)
    return _batch_lpips_difference(img1, img2).item()


def lpips_clustering(images: MultiImagesTyping, threshold: float = 0.45) -> List[int]:
    images = load_images(images, mode='RGB')
    n = len(images)

    feat_list = [lpips_extract_feature(image) for image in tqdm(images, leave=False, desc='Extract features')]
    progress = tqdm(total=n * (n + 1) // 2, leave=False, desc='Metrics')

    @lru_cache(maxsize=n * (n + 1) // 2)
    def _cached_metric(x, y):
        result = lpips_difference(feat_list[x], feat_list[y])
        progress.update(1)
        return result

    def img_sim_metric(x, y):
        x, y = int(min(x, y)), int(max(x, y))
        return _cached_metric(x, y)

    samples = np.array(range(n)).reshape(-1, 1)
    clustering = DBSCAN(eps=threshold, min_samples=2, metric=img_sim_metric).fit(samples)
    progress.close()
    return clustering.labels_
