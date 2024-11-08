"""
Overview:
    A tool used to determine the visual differences between anime characters in two images
    (limited to images containing only a single anime character). When the characters in the two images are the same,
    they should have a smaller difference value.

    The main models of CCIP are trained by `7eu7d7 <https://github.com/7eu7d7>`_,
    and each model along with its corresponding metric data and thresholds are hosted in the
    repository `deepghs/ccip_onnx <https://huggingface.co/deepghs/ccip_onnx>`_.

    This is an overall benchmark of all the operations in CCIP models:

    .. image:: ccip_benchmark.plot.py.svg
        :align: center

    Here are some example images

    .. image:: ccip_small.plot.py.svg
        :align: center

    .. note::
        Due to **significant differences in thresholds and optimal clustering parameters among the CCIP models**,
        it is recommended to refer to the relevant measurement data in the aforementioned
        model repository `deepghs/ccip_onnx <https://huggingface.co/deepghs/ccip_onnx>`_
        before performing any manual operations.
"""
import json
from typing import Literal, Union, List, Optional, Tuple

import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from sklearn.cluster import DBSCAN, OPTICS
from tqdm.auto import tqdm

from ..data import MultiImagesTyping, load_images, ImageTyping
from ..utils import open_onnx_model, ts_lru_cache

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

    'ccip_merge',
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


@ts_lru_cache()
def _open_feat_model(model):
    return open_onnx_model(hf_hub_download(
        f'deepghs/ccip_onnx',
        f'{model}/model_feat.onnx',
    ))


@ts_lru_cache()
def _open_metric_model(model):
    return open_onnx_model(hf_hub_download(
        f'deepghs/ccip_onnx',
        f'{model}/model_metrics.onnx',
    ))


@ts_lru_cache()
def _open_metrics(model):
    with open(hf_hub_download(f'deepghs/ccip_onnx', f'{model}/metrics.json'), 'r') as f:
        return json.load(f)


@ts_lru_cache()
def _open_cluster_metrics(model):
    with open(hf_hub_download(f'deepghs/ccip_onnx', f'{model}/cluster.json'), 'r') as f:
        return json.load(f)


_VALID_MODEL_NAMES = [
    'ccip-caformer-24-randaug-pruned',
    'ccip-caformer-6-randaug-pruned_fp32',
    'ccip-caformer-5_fp32',
]
_DEFAULT_MODEL_NAMES = 'ccip-caformer-24-randaug-pruned'


def ccip_extract_feature(image: ImageTyping, size: int = 384, model: str = _DEFAULT_MODEL_NAMES):
    """
    Extracts the feature vector of the character from the given anime image.

    :param image: The anime image containing a single character.
    :type image: ImageTyping

    :param size: The size of the input image to be used for feature extraction. (default: ``384``)
    :type size: int

    :param model: The name of the model to use for feature extraction. (default: ``ccip-caformer-24-randaug-pruned``)
                  The available model names are: ``ccip-caformer-24-randaug-pruned``,
                  ``ccip-caformer-6-randaug-pruned_fp32``, ``ccip-caformer-5_fp32``.
    :type model: str

    :return: The feature vector of the character.
    :rtype: numpy.ndarray

    Examples::
        >>> from imgutils.metrics import ccip_extract_feature
        >>>
        >>> feat = ccip_extract_feature('ccip/1.jpg')
        >>> feat.shape, feat.dtype
        ((768,), dtype('float32'))
    """
    return ccip_batch_extract_features([image], size, model)[0]


def ccip_batch_extract_features(images: MultiImagesTyping, size: int = 384, model: str = _DEFAULT_MODEL_NAMES):
    """
    Extracts the feature vectors of multiple images using the specified model.

    :param images: The input images from which to extract the feature vectors.
    :type images: MultiImagesTyping

    :param size: The size of the input image to be used for feature extraction. (default: ``384``)
    :type size: int

    :param model: The name of the model to use for feature extraction. (default: ``ccip-caformer-24-randaug-pruned``)
                  The available model names are: ``ccip-caformer-24-randaug-pruned``,
                  ``ccip-caformer-6-randaug-pruned_fp32``, ``ccip-caformer-5_fp32``.
    :type model: str

    :return: The feature vectors of the input images.
    :rtype: numpy.ndarray

    Examples::
        >>> from imgutils.metrics import ccip_batch_extract_features
        >>>
        >>> feat = ccip_batch_extract_features(['ccip/1.jpg', 'ccip/2.jpg', 'ccip/6.jpg'])
        >>> feat.shape, feat.dtype
        ((3, 768), dtype('float32'))
    """
    images = load_images(images, mode='RGB')
    data = np.stack([_preprocess_image(item, size=size) for item in images]).astype(np.float32)
    output, = _open_feat_model(model).run(['output'], {'input': data})
    return output


_FeatureOrImage = Union[ImageTyping, np.ndarray]


def _p_feature(x: _FeatureOrImage, size: int = 384, model: str = _DEFAULT_MODEL_NAMES):
    if isinstance(x, np.ndarray):  # if feature
        return x
    else:  # is image or path
        return ccip_extract_feature(x, size, model)


def ccip_default_threshold(model: str = _DEFAULT_MODEL_NAMES) -> float:
    """
    Retrieves the default threshold value obtained from model metrics in the Hugging Face model repository.

    :param model: The name of the model to use for feature extraction. (default: ``ccip-caformer-24-randaug-pruned``)
                  The available model names are: ``ccip-caformer-24-randaug-pruned``,
                  ``ccip-caformer-6-randaug-pruned_fp32``, ``ccip-caformer-5_fp32``.
    :type model: str

    :return: The default threshold value obtained from model metrics.
    :rtype: float

    Examples::
        >>> from imgutils.metrics import ccip_default_threshold
        >>>
        >>> ccip_default_threshold()
        0.17847511429108218
        >>> ccip_default_threshold('ccip-caformer-6-randaug-pruned_fp32')
        0.1951224011983088
        >>> ccip_default_threshold('ccip-caformer-5_fp32')
        0.18397327797685215
    """
    return _open_metrics(model)['threshold']


def ccip_difference(x: _FeatureOrImage, y: _FeatureOrImage,
                    size: int = 384, model: str = _DEFAULT_MODEL_NAMES) -> float:
    """
    Calculates the difference value between two anime characters based on their images or feature vectors.

    :param x: The image or feature vector of the first anime character.
    :type x: Union[ImageTyping, np.ndarray]

    :param y: The image or feature vector of the second anime character.
    :type y: Union[ImageTyping, np.ndarray]

    :param size: The size of the input image to be used for feature extraction. (default: ``384``)
    :type size: int

    :param model: The name of the model to use for feature extraction. (default: ``ccip-caformer-24-randaug-pruned``)
                  The available model names are: ``ccip-caformer-24-randaug-pruned``,
                  ``ccip-caformer-6-randaug-pruned_fp32``, ``ccip-caformer-5_fp32``.
    :type model: str

    :return: The difference value between the two anime characters.
    :rtype: float

    Examples::
        >>> from imgutils.metrics import ccip_difference
        >>>
        >>> ccip_difference('ccip/1.jpg', 'ccip/2.jpg')  # same character
        0.16583099961280823
        >>>
        >>> # different characters
        >>> ccip_difference('ccip/1.jpg', 'ccip/6.jpg')
        0.42947039008140564
        >>> ccip_difference('ccip/1.jpg', 'ccip/7.jpg')
        0.4037521779537201
        >>> ccip_difference('ccip/2.jpg', 'ccip/6.jpg')
        0.4371533691883087
        >>> ccip_difference('ccip/2.jpg', 'ccip/7.jpg')
        0.40748104453086853
        >>> ccip_difference('ccip/6.jpg', 'ccip/7.jpg')
        0.392294704914093
    """
    return ccip_batch_differences([x, y], size, model)[0, 1].item()


def ccip_same(x: _FeatureOrImage, y: _FeatureOrImage, threshold: Optional[float] = None,
              size: int = 384, model: str = _DEFAULT_MODEL_NAMES) -> float:
    """
    Determines whether two given images or feature vectors belong to the same anime character based on the CCIP model.

    :param x: The image or feature vector of the first anime character.
    :type x: Union[ImageTyping, np.ndarray]

    :param y: The image or feature vector of the second anime character.
    :type y: Union[ImageTyping, np.ndarray]

    :param threshold: The threshold value for determining similarity.
                      If not provided, the default threshold for the model from :func:`ccip_default_threshold` is used.
    :type threshold: Optional[float]

    :param size: The size of the input image to be used for feature extraction. (default: ``384``)
    :type size: int

    :param model: The name of the model to use for feature extraction. (default: ``ccip-caformer-24-randaug-pruned``)
                  The available model names are: ``ccip-caformer-24-randaug-pruned``,
                  ``ccip-caformer-6-randaug-pruned_fp32``, ``ccip-caformer-5_fp32``.
    :type model: str

    :return: True if the images or feature vectors are determined to belong to the same anime character, False otherwise.
    :rtype: bool

    Examples::
        >>> from imgutils.metrics import ccip_same
        >>>
        >>> ccip_same('ccip/1.jpg', 'ccip/2.jpg')  # same character
        True
        >>>
        >>> # different characters
        >>> ccip_same('ccip/1.jpg', 'ccip/6.jpg')
        False
        >>> ccip_same('ccip/1.jpg', 'ccip/7.jpg')
        False
        >>> ccip_same('ccip/2.jpg', 'ccip/6.jpg')
        False
        >>> ccip_same('ccip/2.jpg', 'ccip/7.jpg')
        False
        >>> ccip_same('ccip/6.jpg', 'ccip/7.jpg')
        False
    """
    diff = ccip_difference(x, y, size, model)
    threshold = threshold if threshold is not None else ccip_default_threshold(model)
    return diff <= threshold


def ccip_batch_differences(images: List[_FeatureOrImage],
                           size: int = 384, model: str = _DEFAULT_MODEL_NAMES) -> np.ndarray:
    """
    Calculates the pairwise differences between a given list of images or feature vectors representing anime characters.

    :param images: The list of images or feature vectors representing anime characters.
    :type images: List[Union[ImageTyping, np.ndarray]]

    :param size: The size of the input image to be used for feature extraction. (default: ``384``)
    :type size: int

    :param model: The name of the model to use for feature extraction. (default: ``ccip-caformer-24-randaug-pruned``)
                  The available model names are: ``ccip-caformer-24-randaug-pruned``,
                  ``ccip-caformer-6-randaug-pruned_fp32``, ``ccip-caformer-5_fp32``.
    :type model: str

    :return: The matrix of pairwise differences between the given images or feature vectors.
    :rtype: np.ndarray

    Examples::
        >>> from imgutils.metrics import ccip_batch_differences
        >>>
        >>> ccip_batch_differences(['ccip/1.jpg', 'ccip/2.jpg', 'ccip/6.jpg', 'ccip/7.jpg'])
        array([[6.5350548e-08, 1.6583106e-01, 4.2947042e-01, 4.0375218e-01],
               [1.6583106e-01, 9.8025822e-08, 4.3715334e-01, 4.0748104e-01],
               [4.2947042e-01, 4.3715334e-01, 3.2675274e-08, 3.9229470e-01],
               [4.0375218e-01, 4.0748104e-01, 3.9229470e-01, 6.5350548e-08]],
              dtype=float32)
    """
    input_ = np.stack([_p_feature(img, size, model) for img in images]).astype(np.float32)
    output, = _open_metric_model(model).run(['output'], {'input': input_})
    return output


def ccip_batch_same(images: List[_FeatureOrImage], threshold: Optional[float] = None,
                    size: int = 384, model: str = _DEFAULT_MODEL_NAMES) -> np.ndarray:
    """
    Calculates whether the given list of images or feature vectors representing anime characters are
    the same characters, based on the pairwise differences matrix and a given threshold.

    :param images: The list of images or feature vectors representing anime characters.
    :type images: List[Union[ImageTyping, np.ndarray]]

    :param threshold: The threshold value for determining similarity.
                      If not provided, the default threshold for the model from :func:`ccip_default_threshold` is used.
    :type threshold: Optional[float]

    :param size: The size of the input image to be used for feature extraction. (default: ``384``)
    :type size: int

    :param model: The name of the model to use for feature extraction. (default: ``ccip-caformer-24-randaug-pruned``)
                  The available model names are: ``ccip-caformer-24-randaug-pruned``,
                  ``ccip-caformer-6-randaug-pruned_fp32``, ``ccip-caformer-5_fp32``.
    :type model: str

    :return: A boolean matrix of shape (N, N), where N is the length of the `images` list. The value at position (i, j)
             indicates whether the i-th and j-th characters are considered the same.
    :rtype: np.ndarray

    Examples::
        >>> from imgutils.metrics import ccip_batch_same
        >>>
        >>> ccip_batch_same(['ccip/1.jpg', 'ccip/2.jpg', 'ccip/6.jpg', 'ccip/7.jpg'])
        array([[ True,  True, False, False],
               [ True,  True, False, False],
               [False, False,  True, False],
               [False, False, False,  True]])
    """
    batch_diff = ccip_batch_differences(images, size, model)
    threshold = threshold if threshold is not None else ccip_default_threshold(model)
    return batch_diff <= threshold


CCIPClusterMethodTyping = Literal['dbscan', 'dbscan_2', 'dbscan_free', 'optics', 'optics_best']
_METHOD_MAPPING = {'optics_best': 'optics'}


def ccip_default_clustering_params(model: str = _DEFAULT_MODEL_NAMES,
                                   method: CCIPClusterMethodTyping = 'optics') -> Tuple[float, int]:
    """
    Retrieves the default configuration for clustering operations.

    When the ``method`` is ``dbscan``, the epsilon (eps) value is obtained from
    :func:`ccip_default_threshold` as the default threshold value, and the min_samples value is set to ``2``.

    When the ``method`` is ``optics``, the epsilon (eps) value is set to ``0.5``,
    and the min_samples value is set to ``5``.

    For other values of ``method``, the function automatically retrieves the recommended parameter
    configuration from the optimized models in the HuggingFace model repository.

    :param model: The name of the model to use for feature extraction. (default: ``ccip-caformer-24-randaug-pruned``)
                  The available model names are: ``ccip-caformer-24-randaug-pruned``,
                  ``ccip-caformer-6-randaug-pruned_fp32``, ``ccip-caformer-5_fp32``.
    :type model: str

    :param method: The clustering method for which the default parameters are retrieved.
                   (default: ``optics``)
                   The available options are: ``dbscan``, ``dbscan_2``, ``dbscan_free``, ``optics``, ``optics_best``.
    :type method: CCIPClusterMethodTyping

    :return: A tuple containing the default clustering parameters: (eps, min_samples).
    :rtype: Tuple[float, int]

    Examples::
        >>> from imgutils.metrics import ccip_default_clustering_params
        >>>
        >>> ccip_default_clustering_params()
        (0.5, 5)
        >>> ccip_default_clustering_params(method='dbscan')
        (0.17847511429108218, 2)
        >>> ccip_default_clustering_params(method='dbscan_2')
        (0.12921094122454668, 2)
        >>> ccip_default_clustering_params(method='dbscan_free')
        (0.1291187648928262, 2)
        >>> ccip_default_clustering_params(method='optics_best')
        (0.1836453739562513, 5)
    """
    if method == 'dbscan':
        return ccip_default_threshold(model), 2
    if method == 'optics':
        return 0.5, 5
    else:
        _info = _open_cluster_metrics(model)[_METHOD_MAPPING.get(method, method)]
        return _info['eps'], _info['min_samples']


def ccip_clustering(images: List[_FeatureOrImage], method: CCIPClusterMethodTyping = 'optics',
                    eps: Optional[float] = None, min_samples: Optional[int] = None,
                    size: int = 384, model: str = _DEFAULT_MODEL_NAMES) -> np.ndarray:
    """
    Performs clustering on the given list of images or feature vectors.

    The function applies the selected clustering method (``method``) with
    the specified parameters (``eps`` and ``min_samples``) to cluster the provided list of
    images or feature vectors (``images``).

    The default values for ``eps`` and ``min_samples`` are obtained from
    :func:`ccip_default_clustering_params` based on the selected ``method`` and ``model``.
    If no values are provided for ``eps`` and ``min_samples``, the default values will be used.

    The images or feature vectors are preprocessed and converted to feature representations using
    the specified ``size`` and ``model`` parameters. The pairwise differences between the feature vectors
    are calculated using :func:`ccip_batch_differences` to define the distance metric for clustering.

    The clustering is performed using either the DBSCAN algorithm or the OPTICS algorithm
    based on the selected ``method``.
    The clustering labels are returned as a NumPy array.

    :param images: A list of images or feature vectors to be clustered.
    :type images: List[_FeatureOrImage]

    :param method: The clustering method for which the default parameters are retrieved.
                   (default: ``optics``)
                   The available options are: ``dbscan``, ``dbscan_2``, ``dbscan_free``, ``optics``, ``optics_best``.
    :type method: CCIPClusterMethodTyping

    :param eps: The maximum distance between two samples to be considered in the same neighborhood. (default: ``None``)
                If None, the default value is obtained from :func:`ccip_default_clustering_params`.
    :type eps: Optional[float]

    :param min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
                        (default: ``None``)
                        If None, the default value is obtained from :func:`ccip_default_clustering_params`.
    :type min_samples: Optional[int]

    :param size: The size of the images to be used for feature extraction. (default: ``384``)
    :type size: int

    :param model: The name of the model to use for feature extraction. (default: ``ccip-caformer-24-randaug-pruned``)
                  The available model names are: ``ccip-caformer-24-randaug-pruned``,
                  ``ccip-caformer-6-randaug-pruned_fp32``, ``ccip-caformer-5_fp32``.
    :type model: str

    :return: An array of clustering labels indicating the cluster assignments for each image or feature vector.
    :rtype: np.ndarray

    Examples::
        Here are all the images

        .. image:: ccip_full.plot.py.svg
            :align: center

        >>> from imgutils.metrics import ccip_clustering
        >>>
        >>> images = [f'ccip/{i}.jpg' for i in range(1, 13)]
        >>> images
        ['ccip/1.jpg', 'ccip/2.jpg', 'ccip/3.jpg', 'ccip/4.jpg', 'ccip/5.jpg', 'ccip/6.jpg', 'ccip/7.jpg', 'ccip/8.jpg', 'ccip/9.jpg', 'ccip/10.jpg', 'ccip/11.jpg', 'ccip/12.jpg']
        >>>
        >>> # few images, min_sample should not be too large
        >>> ccip_clustering(images, min_samples=2)
        [0, 0, 0, 3, 3, 3, 1, 1, 1, 1, 2, 2]

    .. note::
        Please note that the clustering process in **CCIP is sensitive to parameters and may require tuning**.
        Therefore, it is recommended to follow these guidelines:

        1. When dealing with a large number of samples, it is recommended to use the default parameters
        of the ``optics`` method for clustering. This helps ensure the robustness of the clustering solution.

        2. If the number of samples is small, it is advised to reduce the value of the ``min_samples`` parameter
        before performing clustering. However, it should be noted that this may significantly increase the possibility
        of separating slightly different instances of the same character into different clusters.

        3. In cases where the samples exhibit a regular pattern overall (e.g., characters with clear
        features and consistent poses and outfits), the ``dbscan`` method can be considered for clustering.
        However, please be aware that the dbscan method is highly sensitive to the ``eps`` value,
        so careful tuning is necessary.

    """
    _default_eps, _default_min_samples = ccip_default_clustering_params(model, method)
    eps = eps or _default_eps
    min_samples = min_samples or _default_min_samples

    images = [_p_feature(img, size, model) for img in tqdm(images, desc='Extract features')]
    batch_diff = ccip_batch_differences(images, size, model)

    def _metric(x, y):
        return batch_diff[int(x), int(y)].item()

    samples = np.arange(len(images)).reshape(-1, 1)
    if 'dbscan' in method:
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric=_metric).fit(samples)
    elif 'optics' in method:
        clustering = OPTICS(max_eps=eps, min_samples=min_samples, metric=_metric).fit(samples)
    else:
        assert False, f'Unknown mode for CCIP clustering - {method!r}.'  # pragma: no cover

    return clustering.labels_.tolist()


def ccip_merge(images: Union[List[_FeatureOrImage], np.ndarray],
               size: int = 384, model: str = _DEFAULT_MODEL_NAMES) -> np.ndarray:
    """
    Merge multiple feature vectors into a single vector.

    :param images: The feature vectors or images to merge.
    :type images: Union[List[_FeatureOrImage], numpy.ndarray]
    :param size: The size of the image. (default: 384)
    :type size: int
    :param model: The name of the model. (default: ``ccip-caformer-24-randaug-pruned``)
    :type model: str
    :return: The merged feature vector.
    :rtype: numpy.ndarray

    Examples::
        >>> from imgutils.metrics import ccip_merge, ccip_batch_differences
        >>>
        >>> images = [f'ccip/{i}.jpg' for i in range(1, 4)]
        >>>
        >>> merged = ccip_merge(images)
        >>> merged.shape
        (768,)
        >>>
        >>> diffs = ccip_batch_differences([merged, *images])[0, 1:]
        >>> diffs
        array([0.07437477, 0.0356068 , 0.04396922], dtype=float32)
        >>> diffs.mean()
        0.05131693
    """
    embs = np.stack([_p_feature(img, size, model) for img in images]).astype(np.float32)
    lengths = np.linalg.norm(embs, axis=-1)
    embs = embs / lengths.reshape(-1, 1)
    ret_embedding = embs.mean(axis=0)
    return ret_embedding / np.linalg.norm(ret_embedding) * lengths.mean()
