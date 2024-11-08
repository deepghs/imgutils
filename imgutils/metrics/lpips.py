"""
Overview:
    Useful utilities based on `richzhang/PerceptualSimilarity <https://github.com/richzhang/PerceptualSimilarity>`_, \
    tested with dataset `deepghs/chafen_arknights(private) <https://huggingface.co/datasets/deepghs/chafen_arknights>`_.

    When threshold is `0.45`, the `adjusted rand score <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html>`_ can reach `0.995`.

    This is an overall benchmark of all the operations in LPIPS models:

    .. image:: lpips_benchmark.plot.py.svg
        :align: center

"""
from functools import lru_cache
from typing import Tuple, Union, List

import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from sklearn.cluster import DBSCAN
from tqdm.auto import tqdm

from imgutils.data import rgb_encode, MultiImagesTyping, load_images, ImageTyping, load_image
from imgutils.utils import open_onnx_model, ts_lru_cache

__all__ = [
    'lpips_extract_feature',
    'lpips_difference',
    'lpips_clustering',
]


def _image_resize(image: Image.Image, size=400):
    return image.resize((size, size), resample=Image.BILINEAR)


def _image_encode(image: Image.Image):
    return rgb_encode(_image_resize(image))


@ts_lru_cache()
def _lpips_feature_model():
    return open_onnx_model(hf_hub_download(
        'deepghs/imgutils-models',
        'lpips/lpips_feature.onnx',
    ))


def lpips_extract_feature(image: MultiImagesTyping) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Overview:
        Extract feature from images.

    :param image: One or multiple images to extract features.
    :return: Extracted features, should be a tuple of 5 arrays which extracted from CNN.

    Example:
        >>> from imgutils.metrics import lpips_extract_feature
        >>>
        >>> f1, f2, f3, f4, f5 = lpips_extract_feature('lpips/1.jpg')
        >>> (f1.shape, f2.shape, f3.shape, f4.shape, f4.shape)
        ((1, 64, 99, 99), (1, 192, 49, 49), (1, 384, 24, 24), (1, 256, 24, 24), (1, 256, 24, 24))
        >>> f1, f2, f3, f4, f5 = lpips_extract_feature(['lpips/1.jpg', 'lpips/4.jpg', 'lpips/7.jpg'])
        >>> (f1.shape, f2.shape, f3.shape, f4.shape, f4.shape)
        ((3, 64, 99, 99), (3, 192, 49, 49), (3, 384, 24, 24), (3, 256, 24, 24), (3, 256, 24, 24))

    .. note::
        This feature can be used in :func:`lpips_difference` and :func:`lpips_clustering`.
    """
    images = load_images(image)
    _encoded = np.stack([_image_encode(image) for image in images])
    features = _lpips_feature_model().run(["feat_0", "feat_1", "feat_2", "feat_3", "feat_4"], {'input': _encoded})
    return tuple(features)


@ts_lru_cache()
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
    """
    Overview:
        Calculate LPIPS difference between images.

    :param img1: Image file, PIL object or extracted feature of image.
    :param img2: Image file, PIL object or extracted feature of another image.
    :return: LPIPS difference. Value lower than ``0.45`` usually represents similar.

    Example:
        Here are some images for example

        .. image:: lpips_small.plot.py.svg
           :align: center

        >>> from imgutils.metrics import lpips_difference
        >>>
        >>> lpips_difference('lpips/1.jpg', 'lpips/2.jpg')
        0.16922694444656372
        >>> lpips_difference('lpips/1.jpg', 'lpips/3.jpg')
        0.22250649333000183
        >>> lpips_difference('lpips/1.jpg', 'lpips/4.jpg')  # not similar
        0.6897575259208679
        >>> lpips_difference('lpips/2.jpg', 'lpips/3.jpg')
        0.10956494510173798
        >>> lpips_difference('lpips/2.jpg', 'lpips/4.jpg')  # not similar
        0.6823137998580933
        >>> lpips_difference('lpips/3.jpg', 'lpips/4.jpg')  # not similar
        0.6837796568870544
    """
    img1 = _auto_feat(img1)
    img2 = _auto_feat(img2)
    return _batch_lpips_difference(img1, img2).item()


def lpips_clustering(images: MultiImagesTyping, threshold: float = 0.45) -> List[int]:
    """
    Overview:
        Clustering images with LPIPS metrics.

    :param images: List of multiple images.
    :param threshold: Threshold of clustering. Default value ``0.45`` is recommended.
    :return: Clustering result with LPIPS, each integer represent one group, ``-1`` means this is a noise sample.

    Example:
        Here are some images for example

        .. image:: lpips_full.plot.py.svg
           :align: center

        >>> from imgutils.metrics import lpips_clustering
        >>>
        >>> images = [f'lpips/{i}.jpg' for i in range(1, 10)]
        >>> images
        ['lpips/1.jpg', 'lpips/2.jpg', 'lpips/3.jpg', 'lpips/4.jpg', 'lpips/5.jpg', 'lpips/6.jpg', 'lpips/7.jpg', 'lpips/8.jpg', 'lpips/9.jpg']
        >>> lpips_clustering(images)
        [0, 0, 0, 1, 1, -1, -1, -1, -1]
    """
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
    return clustering.labels_.tolist()
