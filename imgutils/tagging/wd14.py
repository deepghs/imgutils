"""
Overview:
    This module provides utilities for image tagging using WD14 taggers.
    It includes functions for loading models, processing images, and extracting tags.

    The module is inspired by the `SmilingWolf/wd-v1-4-tags <https://huggingface.co/spaces/SmilingWolf/wd-v1-4-tags>`_
    project on Hugging Face.

"""
from typing import List, Tuple

import numpy as np
import onnxruntime
import pandas as pd
from PIL import Image
from hbutils.testing.requires.version import VersionInfo
from huggingface_hub import hf_hub_download

from .format import remove_underline
from .overlap import drop_overlap_tags
from ..data import load_image, ImageTyping
from ..utils import open_onnx_model, vreplace, sigmoid, ts_lru_cache

SWIN_MODEL_REPO = "SmilingWolf/wd-v1-4-swinv2-tagger-v2"
CONV_MODEL_REPO = "SmilingWolf/wd-v1-4-convnext-tagger-v2"
CONV2_MODEL_REPO = "SmilingWolf/wd-v1-4-convnextv2-tagger-v2"
VIT_MODEL_REPO = "SmilingWolf/wd-v1-4-vit-tagger-v2"
MOAT_MODEL_REPO = "SmilingWolf/wd-v1-4-moat-tagger-v2"
CONV_V3_MODEL_REPO = 'SmilingWolf/wd-convnext-tagger-v3'
SWIN_V3_MODEL_REPO = 'SmilingWolf/wd-swinv2-tagger-v3'
VIT_V3_MODEL_REPO = 'SmilingWolf/wd-vit-tagger-v3'
VIT_LARGE_MODEL_REPO = 'SmilingWolf/wd-vit-large-tagger-v3'
EVA02_LARGE_MODEL_DSV3_REPO = "SmilingWolf/wd-eva02-large-tagger-v3"
MODEL_FILENAME = "model.onnx"
LABEL_FILENAME = "selected_tags.csv"

_IS_V3_SUPPORT = VersionInfo(onnxruntime.__version__) >= '1.17'

MODEL_NAMES = {
    "EVA02_Large": EVA02_LARGE_MODEL_DSV3_REPO,
    "ViT_Large": VIT_LARGE_MODEL_REPO,

    "SwinV2": SWIN_MODEL_REPO,
    "ConvNext": CONV_MODEL_REPO,
    "ConvNextV2": CONV2_MODEL_REPO,
    "ViT": VIT_MODEL_REPO,
    "MOAT": MOAT_MODEL_REPO,

    "SwinV2_v3": SWIN_V3_MODEL_REPO,
    "ConvNext_v3": CONV_V3_MODEL_REPO,
    "ViT_v3": VIT_V3_MODEL_REPO,
}
_DEFAULT_MODEL_NAME = 'SwinV2_v3'


def _version_support_check(model_name):
    """
    Check if the current onnxruntime version supports the given model.

    :param model_name: The name of the model to check.
    :type model_name: str
    :raises EnvironmentError: If the model is not supported by the current onnxruntime version.
    """
    if model_name.endswith('_v3') and not _IS_V3_SUPPORT:
        raise EnvironmentError(f'V3 taggers not supported on onnxruntime {onnxruntime.__version__}, '
                               f'please upgrade it to 1.17+ version.\n'
                               f'If you are running on CPU, use "pip install -U onnxruntime" .\n'
                               f'If you are running on GPU, use "pip install -U onnxruntime-gpu" .')  # pragma: no cover


@ts_lru_cache()
def _get_wd14_model(model_name):
    """
    Load an ONNX model from the Hugging Face Hub.

    :param model_name: The name of the model to load.
    :type model_name: str
    :return: The loaded ONNX model.
    :rtype: ONNXModel
    """
    _version_support_check(model_name)
    return open_onnx_model(hf_hub_download(
        repo_id='deepghs/wd14_tagger_with_embeddings',
        filename=f'{MODEL_NAMES[model_name]}/model.onnx',
    ))


@ts_lru_cache()
def _get_wd14_weights(model_name):
    """
    Load the weights for a WD14 model.

    :param model_name: The name of the model.
    :type model_name: str
    :return: The loaded weights.
    :rtype: numpy.ndarray
    """
    _version_support_check(model_name)
    return np.load(hf_hub_download(
        repo_id='deepghs/wd14_tagger_with_embeddings',
        filename=f'{MODEL_NAMES[model_name]}/inv.npz',
    ))


@ts_lru_cache()
def _get_wd14_labels(model_name, no_underline: bool = False) -> Tuple[List[str], List[int], List[int], List[int]]:
    """
    Get labels for the WD14 model.

    :param model_name: The name of the model.
    :type model_name: str
    :param no_underline: If True, replaces underscores in tag names with spaces.
    :type no_underline: bool
    :return: A tuple containing the list of tag names, and lists of indexes for rating, general, and character categories.
    :rtype: Tuple[List[str], List[int], List[int], List[int]]
    """
    path = hf_hub_download(MODEL_NAMES[model_name], LABEL_FILENAME)
    df = pd.read_csv(path)
    name_series = df["name"]
    if no_underline:
        name_series = name_series.map(remove_underline)
    tag_names = name_series.tolist()

    rating_indexes = list(np.where(df["category"] == 9)[0])
    general_indexes = list(np.where(df["category"] == 0)[0])
    character_indexes = list(np.where(df["category"] == 4)[0])
    return tag_names, rating_indexes, general_indexes, character_indexes


def _mcut_threshold(probs) -> float:
    """
    Compute the Maximum Cut Thresholding (MCut) for multi-label classification.

    This method is based on the paper:
    Largeron, C., Moulin, C., & Gery, M. (2012). MCut: A Thresholding Strategy
    for Multi-label Classification. In 11th International Symposium, IDA 2012
    (pp. 172-183).

    :param probs: Array of probabilities.
    :type probs: numpy.ndarray
    :return: The computed threshold.
    :rtype: float
    """
    sorted_probs = probs[probs.argsort()[::-1]]
    difs = sorted_probs[:-1] - sorted_probs[1:]
    t = difs.argmax()
    thresh = (sorted_probs[t] + sorted_probs[t + 1]) / 2
    return thresh


def _prepare_image_for_tagging(image: ImageTyping, target_size: int):
    """
    Prepare an image for tagging by resizing and padding it.

    :param image: The input image.
    :type image: ImageTyping
    :param target_size: The target size for the image.
    :type target_size: int
    :return: The prepared image as a numpy array.
    :rtype: numpy.ndarray
    """
    image = load_image(image, force_background=None, mode=None)
    image_shape = image.size
    max_dim = max(image_shape)
    pad_left = (max_dim - image_shape[0]) // 2
    pad_top = (max_dim - image_shape[1]) // 2

    padded_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
    try:
        padded_image.paste(image, (pad_left, pad_top), mask=image)
    except ValueError:
        padded_image.paste(image, (pad_left, pad_top))

    if max_dim != target_size:
        padded_image = padded_image.resize((target_size, target_size), Image.BICUBIC)

    image_array = np.asarray(padded_image, dtype=np.float32)
    image_array = image_array[:, :, ::-1]
    return np.expand_dims(image_array, axis=0)


def _postprocess_embedding(
        pred, embedding,
        model_name: str = _DEFAULT_MODEL_NAME,
        general_threshold: float = 0.35,
        general_mcut_enabled: bool = False,
        character_threshold: float = 0.85,
        character_mcut_enabled: bool = False,
        no_underline: bool = False,
        drop_overlap: bool = False,
        fmt=('rating', 'general', 'character'),
):
    """
    Post-process the embedding and prediction results.

    :param pred: The prediction array.
    :type pred: numpy.ndarray
    :param embedding: The embedding array.
    :type embedding: numpy.ndarray
    :param model_name: The name of the model used.
    :type model_name: str
    :param general_threshold: Threshold for general tags.
    :type general_threshold: float
    :param general_mcut_enabled: Whether to use MCut for general tags.
    :type general_mcut_enabled: bool
    :param character_threshold: Threshold for character tags.
    :type character_threshold: float
    :param character_mcut_enabled: Whether to use MCut for character tags.
    :type character_mcut_enabled: bool
    :param no_underline: Whether to remove underscores from tag names.
    :type no_underline: bool
    :param drop_overlap: Whether to drop overlapping tags.
    :type drop_overlap: bool
    :param fmt: The format of the output.
    :return: The post-processed results.
    """
    assert len(pred.shape) == len(embedding.shape) == 1, \
        f'Both pred and embeddings shapes should be 1-dim, ' \
        f'but pred: {pred.shape!r}, embedding: {embedding.shape!r} actually found.'
    tag_names, rating_indexes, general_indexes, character_indexes = _get_wd14_labels(model_name, no_underline)
    labels = list(zip(tag_names, pred.astype(float)))

    rating = {labels[i][0]: labels[i][1].item() for i in rating_indexes}

    general_names = [labels[i] for i in general_indexes]
    if general_mcut_enabled:
        general_probs = np.array([x[1] for x in general_names])
        general_threshold = _mcut_threshold(general_probs)

    general_res = {x: v.item() for x, v in general_names if v > general_threshold}
    if drop_overlap:
        general_res = drop_overlap_tags(general_res)

    character_names = [labels[i] for i in character_indexes]
    if character_mcut_enabled:
        character_probs = np.array([x[1] for x in character_names])
        character_threshold = _mcut_threshold(character_probs)
        character_threshold = max(0.15, character_threshold)

    character_res = {x: v.item() for x, v in character_names if v > character_threshold}

    return vreplace(
        fmt,
        {
            'rating': rating,
            'general': general_res,
            'character': character_res,
            'tag': {**general_res, **character_res},
            'embedding': embedding.astype(np.float32),
            'prediction': pred.astype(np.float32),
        }
    )


def get_wd14_tags(
        image: ImageTyping,
        model_name: str = _DEFAULT_MODEL_NAME,
        general_threshold: float = 0.35,
        general_mcut_enabled: bool = False,
        character_threshold: float = 0.85,
        character_mcut_enabled: bool = False,
        no_underline: bool = False,
        drop_overlap: bool = False,
        fmt=('rating', 'general', 'character'),
):
    """
    Get tags for an image using WD14 taggers.

    This function is similar to the
    `SmilingWolf/wd-v1-4-tags <https://huggingface.co/spaces/SmilingWolf/wd-v1-4-tags>`_ project on Hugging Face.

    :param image: The input image.
    :type image: ImageTyping
    :param model_name: The name of the model to use.
    :type model_name: str
    :param general_threshold: The threshold for general tags.
    :type general_threshold: float
    :param general_mcut_enabled: If True, applies MCut thresholding to general tags.
    :type general_mcut_enabled: bool
    :param character_threshold: The threshold for character tags.
    :type character_threshold: float
    :param character_mcut_enabled: If True, applies MCut thresholding to character tags.
    :type character_mcut_enabled: bool
    :param no_underline: If True, replaces underscores in tag names with spaces.
    :type no_underline: bool
    :param drop_overlap: If True, drops overlapping tags.
    :type drop_overlap: bool
    :param fmt: Return format, default is ``('rating', 'general', 'character')``.
        ``embedding`` is also supported for feature extraction.
    :return: Prediction result based on the provided fmt.

    .. note::
        The fmt argument can include the following keys:

        - ``rating``: a dict containing ratings and their confidences
        - ``general``: a dict containing general tags and their confidences
        - ``character``: a dict containing character tags and their confidences
        - ``tag``: a dict containing all tags (including general and character, not including rating) and their confidences
        - ``embedding``: a 1-dim embedding of image, recommended for index building after L2 normalization
        - ``prediction``: a 1-dim prediction result of image

        You can extract embedding of the given image with the follwing code

        >>> from imgutils.tagging import get_wd14_tags
        >>>
        >>> embedding = get_wd14_tags('skadi.jpg', fmt='embedding')
        >>> embedding.shape
        (1024, )

        This embedding is valuable for constructing indices that enable rapid querying of images based on
        visual features within large-scale datasets.

    Example:
        Here are some images for example

        .. image:: tagging_demo.plot.py.svg
           :align: center

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
    _, target_size, _, _ = model.get_inputs()[0].shape
    image = _prepare_image_for_tagging(image, target_size)

    input_name = model.get_inputs()[0].name
    assert len(model.get_outputs()) == 2
    label_name = model.get_outputs()[0].name
    emb_name = model.get_outputs()[1].name
    preds, embeddings = model.run([label_name, emb_name], {input_name: image})

    return _postprocess_embedding(
        pred=preds[0],
        embedding=embeddings[0],
        model_name=model_name,
        general_threshold=general_threshold,
        general_mcut_enabled=general_mcut_enabled,
        character_threshold=character_threshold,
        character_mcut_enabled=character_mcut_enabled,
        no_underline=no_underline,
        drop_overlap=drop_overlap,
        fmt=fmt,
    )


_DEFAULT_DENORMALIZER_NAME = 'mnum2_all'


def convert_wd14_emb_to_prediction(
        emb: np.ndarray,
        model_name: str = _DEFAULT_MODEL_NAME,
        general_threshold: float = 0.35,
        general_mcut_enabled: bool = False,
        character_threshold: float = 0.85,
        character_mcut_enabled: bool = False,
        no_underline: bool = False,
        drop_overlap: bool = False,
        fmt=('rating', 'general', 'character'),
        denormalize: bool = False,
        denormalizer_name: str = _DEFAULT_DENORMALIZER_NAME,
):
    """
    Convert WD14 embedding to understandable prediction result. This function can process both
    single embeddings (1-dimensional array) and batches of embeddings (2-dimensional array).

    :param emb: The extracted embedding(s). Can be either a 1-dim array for single image or
                2-dim array for batch processing
    :type emb: numpy.ndarray
    :param model_name: Name of the WD14 model to use for prediction
    :type model_name: str
    :param general_threshold: Confidence threshold for general tags (0.0 to 1.0)
    :type general_threshold: float
    :param general_mcut_enabled: Enable MCut thresholding for general tags to improve prediction quality
    :type general_mcut_enabled: bool
    :param character_threshold: Confidence threshold for character tags (0.0 to 1.0)
    :type character_threshold: float
    :param character_mcut_enabled: Enable MCut thresholding for character tags to improve prediction quality
    :type character_mcut_enabled: bool
    :param no_underline: Replace underscores with spaces in tag names for better readability
    :type no_underline: bool
    :param drop_overlap: Remove overlapping tags to reduce redundancy
    :type drop_overlap: bool
    :param fmt: Specify return format structure for predictions, default is ``('rating', 'general', 'character')``.
    :type fmt: tuple
    :param denormalize: Whether to denormalize the embedding before prediction
    :type denormalize: bool
    :param denormalizer_name: Name of the denormalizer to use if denormalization is enabled
    :type denormalizer_name: str
    :return: For single embeddings: prediction result based on fmt. For batches: list of prediction results.

    .. note::
        Only the embeddings not get normalized can be converted to understandable prediction result.
        If normalized embeddings are provided, set ``denormalize=True`` to convert them back.

    For batch processing (2-dim input), returns a list where each element corresponds
    to one embedding's predictions in the same format as single embedding output.

    Example:
        >>> import os
        >>> import numpy as np
        >>> from imgutils.tagging import get_wd14_tags, convert_wd14_emb_to_prediction
        >>>
        >>> # extract the feature embedding, shape: (W, )
        >>> embedding = get_wd14_tags('skadi.jpg', fmt='embedding')
        >>>
        >>> # convert to understandable result
        >>> rating, general, character = convert_wd14_emb_to_prediction(embedding)
        >>> # these 3 dicts will be the same as that returned by `get_wd14_tags('skadi.jpg')`
        >>>
        >>> # Batch processing, shape: (B, W)
        >>> embeddings = np.stack([
        ...     get_wd14_tags('img1.jpg', fmt='embedding'),
        ...     get_wd14_tags('img2.jpg', fmt='embedding'),
        ... ])
        >>> # results will be a list of (rating, general, character) tuples
        >>> results = convert_wd14_emb_to_prediction(embeddings)
    """
    if denormalize:
        emb = denormalize_wd14_emb(
            emb=emb,
            model_name=model_name,
            denormalizer_name=denormalizer_name,
        )

    z_weights = _get_wd14_weights(model_name)
    weights, bias = z_weights['weights'], z_weights['bias']
    pred = sigmoid(emb @ weights + bias)
    if len(emb.shape) == 1:
        return _postprocess_embedding(
            pred=pred,
            embedding=emb,
            model_name=model_name,
            general_threshold=general_threshold,
            general_mcut_enabled=general_mcut_enabled,
            character_threshold=character_threshold,
            character_mcut_enabled=character_mcut_enabled,
            no_underline=no_underline,
            drop_overlap=drop_overlap,
            fmt=fmt,
        )
    else:
        return [
            _postprocess_embedding(
                pred=pred_item,
                embedding=emb_item,
                model_name=model_name,
                general_threshold=general_threshold,
                general_mcut_enabled=general_mcut_enabled,
                character_threshold=character_threshold,
                character_mcut_enabled=character_mcut_enabled,
                no_underline=no_underline,
                drop_overlap=drop_overlap,
                fmt=fmt,
            )
            for pred_item, emb_item in zip(pred, emb)
        ]


@ts_lru_cache()
def _open_denormalize_model(
        model_name: str = _DEFAULT_MODEL_NAME,
        denormalizer_name: str = _DEFAULT_DENORMALIZER_NAME,
):
    """
    Open a denormalization model for WD14 embeddings.

    :param model_name: Name of the model.
    :type model_name: str
    :param denormalizer_name: Name of the denormalizer.
    :type denormalizer_name: str
    :return: The loaded ONNX model.
    :rtype: ONNXModel
    """
    return open_onnx_model(hf_hub_download(
        repo_id='deepghs/embedding_aligner',
        repo_type='model',
        filename=f'{model_name}_{denormalizer_name}/model.onnx',
    ))


def denormalize_wd14_emb(
        emb: np.ndarray,
        model_name: str = _DEFAULT_MODEL_NAME,
        denormalizer_name: str = _DEFAULT_DENORMALIZER_NAME,
) -> np.ndarray:
    """
    Denormalize WD14 embeddings.

    :param emb: The embedding to denormalize.
    :type emb: numpy.ndarray
    :param model_name: Name of the model.
    :type model_name: str
    :param denormalizer_name: Name of the denormalizer.
    :type denormalizer_name: str
    :return: The denormalized embedding.
    :rtype: numpy.ndarray

    Examples:
        >>> import numpy as np
        >>> from imgutils.tagging import get_wd14_tags, convert_wd14_emb_to_prediction, denormalize_wd14_emb
        ...
        >>> embedding, (r, g, c) = get_wd14_tags(
        ...     'image.png',
        ...     fmt=('embedding', ('rating', 'general', 'character')),
        ... )
        ...
        >>> # normalize embedding
        >>> embedding = embedding / np.linalg.norm(embedding)
        ...
        >>> # denormalize this embedding
        >>> output = denormalize_wd14_emb(embedding)
        ...
        >>> # should be similar to r, g, c, approx 1e-3 error
        >>> rating, general, character = convert_wd14_emb_to_prediction(output)
    """
    model = _open_denormalize_model(
        model_name=model_name,
        denormalizer_name=denormalizer_name,
    )
    emb = emb / np.linalg.norm(emb, axis=-1, keepdims=True)
    if len(emb.shape) == 1:
        output, = model.run(output_names=['embedding'], input_feed={'input': emb[None, ...]})
        return output[0]
    else:
        embedding_width = model.get_outputs()[0].shape[-1]
        origin_shape = emb.shape
        emb = emb.reshape(-1, embedding_width)
        output, = model.run(output_names=['embedding'], input_feed={'input': emb})
        return output.reshape(*origin_shape)
