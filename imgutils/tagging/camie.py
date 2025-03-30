"""
Overview:
    This module provides functionality for image tagging using the Camie Tagger model,
    which can identify over 70,000 tags in images.
    The implementation is based on the `Camais03/camie-tagger <https://huggingface.co/Camais03/camie-tagger>`_ project,
    with ONNX optimizations available at `deepghs/camie_tagger_onnx <https://huggingface.co/deepghs/camie_tagger_onnx>`_.

.. note::
    The tagger categorizes tags into multiple types including rating, general, characters, year, meta, artist,
    and copyright. While rating, general, and character tags tend to be accurate, other tag types
    (year, meta, artist, copyright) have shown limited accuracy in testing and are not included in default outputs.
"""
import json
from typing import List, Tuple, Dict, Optional, Union, Literal, Any

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

from .format import remove_underline
from .overlap import drop_overlap_tags
from ..data import ImageTyping, load_image
from ..preprocess import create_pillow_transforms
from ..utils import open_onnx_model, ts_lru_cache, vnames, vreplace

_REPO_ID = 'deepghs/camie_tagger_onnx'
_DEFAULT_MODEL_NAME = 'initial'
_CATEGORY_MAPS = {
    'rating': 9,
    'year': 10,
    'general': 0,
    'artist': 1,
    'copyright': 3,
    'character': 4,
    'meta': 5,
}


@ts_lru_cache()
def _get_camie_model(model_name, is_full: bool = True):
    """
    Load and cache a Camie ONNX model from the Hugging Face Hub.

    :param model_name: Name of the model to load
    :type model_name: str
    :param is_full: Whether to load the full model or initial-only version
    :type is_full: bool
    :return: Loaded ONNX model
    :rtype: ONNXModel
    """
    return open_onnx_model(hf_hub_download(
        repo_id=_REPO_ID,
        repo_type='model',
        filename=f'{model_name}/model.onnx' if is_full else f'{model_name}/model_initial_only.onnx',
    ))


@ts_lru_cache()
def _get_camie_labels(model_name, no_underline: bool = False) -> Tuple[List[str], Dict[str, List[int]]]:
    """
    Retrieve and process labels for the Camie model.

    :param model_name: Name of the model
    :type model_name: str
    :param no_underline: If True, replace underscores with spaces in tag names
    :type no_underline: bool
    :return: Tuple of (list of tag names, dictionary mapping category names to their indices)
    :rtype: Tuple[List[str], Dict[str, List[int]]]
    """
    path = hf_hub_download(
        repo_id=_REPO_ID,
        repo_type='model',
        filename=f'{model_name}/selected_tags.csv',
    )
    df = pd.read_csv(path)
    name_series = df["name"]
    if no_underline:
        name_series = name_series.map(remove_underline)
    tag_names: List[str] = name_series.tolist()

    indices = {
        cate_name: list(np.where(df["category"] == category)[0])
        for cate_name, category in _CATEGORY_MAPS.items()
    }
    return tag_names, indices


@ts_lru_cache()
def _get_camie_preprocessor(model_name: str):
    """
    Get the image preprocessor for the specified Camie model.

    :param model_name: Name of the model
    :type model_name: str
    :return: Pillow transform pipeline for image preprocessing
    :rtype: callable
    """
    with open(hf_hub_download(
            repo_id=_REPO_ID,
            repo_type='model',
            filename=f'{model_name}/preprocess.json',
    ), 'r') as f:
        p = json.load(f)['stages']
        return create_pillow_transforms(p)


CamieModeTyping = Literal['balanced', 'high_precision', 'high_recall', 'micro_opt', 'macro_opt']


@ts_lru_cache()
def _get_camie_threshold(model_name: str, mode: CamieModeTyping = 'balanced'):
    """
    Get threshold values for different categories based on the specified mode.

    :param model_name: Name of the model
    :type model_name: str
    :param mode: Prediction mode affecting threshold values
    :type mode: CamieModeTyping
    :return: Dictionary of thresholds for each category
    :rtype: Dict[str, float]
    """
    with open(hf_hub_download(
            repo_id=_REPO_ID,
            repo_type='model',
            filename=f'{model_name}/threshold.json',
    ), 'r') as f:
        raw = json.load(f)
        return {
            cate_name: raw[cate_name][mode]['threshold']
            for cate_name in _CATEGORY_MAPS.keys()
        }


def _postprocess_embedding_values(
        pred, logits, embedding,
        model_name: str = _DEFAULT_MODEL_NAME,
        mode: CamieModeTyping = 'balanced',
        thresholds: Optional[Union[float, Dict[str, float]]] = None,
        no_underline: bool = False,
        drop_overlap: bool = False,
):
    """
    Post-process model predictions and embeddings into structured tag results.

    :param pred: Raw prediction array from the model
    :type pred: numpy.ndarray
    :param logits: Logits array from the model
    :type logits: numpy.ndarray
    :param embedding: Embedding array from the model
    :type embedding: numpy.ndarray
    :param model_name: Name of the model used
    :type model_name: str
    :param mode: Prediction mode affecting threshold values
    :type mode: CamieModeTyping
    :param thresholds: Custom thresholds for tag selection
    :type thresholds: Optional[Union[float, Dict[str, float]]]
    :param no_underline: Whether to remove underscores from tag names
    :type no_underline: bool
    :param drop_overlap: Whether to remove overlapping tags
    :type drop_overlap: bool
    :return: Dictionary containing processed predictions and embeddings
    :rtype: Dict[str, Any]
    """
    assert len(pred.shape) == len(embedding.shape) == 1, \
        f'Both pred and embeddings shapes should be 1-dim, ' \
        f'but pred: {pred.shape!r}, embedding: {embedding.shape!r} actually found.'
    tag_names, indices = _get_camie_labels(model_name, no_underline)
    labels = list(zip(tag_names, pred.astype(float)))

    default_thresholds = _get_camie_threshold(model_name=model_name, mode=mode)

    rating = {labels[i][0]: labels[i][1].item() for i in indices['rating']}
    tags, d_tags = {}, {}

    for cate_name, index in indices.items():
        if cate_name == 'rating':
            continue

        if thresholds is not None:
            if isinstance(thresholds, dict):
                threshold = thresholds.get(cate_name, default_thresholds[cate_name])
            elif isinstance(thresholds, (int, float)):
                threshold = thresholds
            else:
                raise TypeError(f'Unknown thresholds type for camie tagger - {thresholds!r}.')
        else:
            threshold = default_thresholds[cate_name]

        names = [labels[i] for i in index]
        cate_pred = {x: v.item() for x, v in names if v >= threshold}
        if cate_name == 'general' and drop_overlap:
            cate_pred = drop_overlap_tags(cate_pred)
        tags.update(cate_pred)
        d_tags[cate_name] = cate_pred

    return {
        'rating': rating,
        **d_tags,
        'tag': tags,
        'embedding': embedding.astype(np.float32),
        'logits': logits.astype(np.float32),
        'prediction': pred.astype(np.float32),
    }


def get_camie_tags(
        image: ImageTyping,
        model_name: str = _DEFAULT_MODEL_NAME,
        mode: CamieModeTyping = 'balanced',
        thresholds: Optional[Union[float, Dict[str, float]]] = None,
        no_underline: bool = False,
        drop_overlap: bool = False,
        fmt: Any = ('rating', 'general', 'character'),
):
    """
    Extract tags from an image using the Camie model.

    :param image: Input image (can be path, URL, or image data).
    :type image: ImageTyping
    :param model_name: Name of the Camie model to use.
    :type model_name: str
    :param mode: Prediction mode affecting threshold values.
    :type mode: CamieModeTyping
    :param thresholds: Custom thresholds for tag selection.
    :type thresholds: Optional[Union[float, Dict[str, float]]]
    :param no_underline: Whether to remove underscores from tag names. Default is ``False``.
    :type no_underline: bool
    :param drop_overlap: Whether to remove overlapping tags. Default is ``False``.
    :type drop_overlap: bool
    :param fmt: Format specification for output. Default is ``('rating', 'general', 'character')``.
    :type fmt: Any
    :return: Extracted tags and embeddings, follow the format from ``fmt``.
    :rtype: Any

    .. note::
        Modes for selection:

        - ``balanced``: Balanced precision/recall
        - ``high_precision``: Higher precision thresholds
        - ``high_recall``: Higher recall thresholds
        - ``micro_opt``: Micro-optimized thresholds
        - ``macro_opt``: Macro-optimized thresholds

    .. note::
        The fmt argument can include the following keys:

        - ``rating``: a dict containing ratings and their confidences
        - ``general``: a dict containing general tags and their confidences
        - ``character``: a dict containing character tags and their confidences
        - ``copyright``: a dict containing copyright tags and their confidences
        - ``artist``: a dict containing artist tags and their confidences
        - ``meta``: a dict containing meta tags and their confidences
        - ``year``: a dict containing year tags and their confidences
        - ``tag``: a dict containing all tags (including general, character, copyright, artist, meta, year, not including rating) and their confidences
        - ``embedding``: a 1-dim embedding of image, recommended for index building after L2 normalization
        - ``prediction``: a 1-dim prediction result of image

        You can extract embedding of the given image with the following code

        >>> from imgutils.tagging import get_camie_tags
        >>>
        >>> embedding = get_camie_tags('skadi.jpg', fmt='embedding')
        >>> embedding.shape
        (1280, )

        This embedding is valuable for constructing indices that enable rapid querying of images based on
        visual features within large-scale datasets.

    .. warning::
        From our testings, other tag types (year, meta, artist, copyright) have shown limited accuracy.
        Especially for artist tags, just a bit better than ``np.random.randn``. So these tag types are
        not included in default ``fmt``.

    Example:
        Here are some images for example

        .. image:: tagging_demo.plot.py.svg
           :align: center

        >>> rating, features, chars = get_camie_tags('skadi.jpg')
        >>> rating
        {'general': 0.04246556758880615, 'sensitive': 0.6936423778533936, 'questionable': 0.23721203207969666, 'explicit': 0.033293724060058594}
        >>> features
        {'1girl': 0.8412569165229797, 'blush': 0.38029077649116516, 'breasts': 0.618192195892334, 'cowboy_shot': 0.37446439266204834, 'large_breasts': 0.5698797702789307, 'long_hair': 0.7119565010070801, 'looking_at_viewer': 0.5252856612205505, 'shirt': 0.46417444944381714, 'solo': 0.5428758859634399, 'standing': 0.34731733798980713, 'tail': 0.3911612927913666, 'thigh_gap': 0.2932726740837097, 'thighs': 0.4544200003147125, 'very_long_hair': 0.44711941480636597, 'ass': 0.2854885458946228, 'outdoors': 0.6344638466835022, 'red_eyes': 0.611354410648346, 'day': 0.564970850944519, 'hair_between_eyes': 0.4444340467453003, 'holding': 0.35846662521362305, 'parted_lips': 0.3867686092853546, 'blue_sky': 0.3723931908607483, 'cloud': 0.31086698174476624, 'short_sleeves': 0.43279752135276794, 'sky': 0.3896197974681854, 'gloves': 0.6638736724853516, 'grey_hair': 0.5094802975654602, 'sweat': 0.4867050349712372, 'navel': 0.6593714952468872, 'crop_top': 0.5243107676506042, 'shorts': 0.4374789893627167, 'artist_name': 0.3754707872867584, 'midriff': 0.6238733530044556, 'ass_visible_through_thighs': 0.31088054180145264, 'gym_uniform': 0.37657681107521057, 'black_shirt': 0.3012588620185852, 'watermark': 0.5147127509117126, 'web_address': 0.6296812295913696, 'short_shorts': 0.29214906692504883, 'black_shorts': 0.37801358103752136, 'buruma': 0.536261260509491, 'bike_shorts': 0.35828399658203125, 'black_gloves': 0.4156728982925415, 'sportswear': 0.44427722692489624, 'baseball_bat': 0.2838006019592285, 'crop_top_overhang': 0.49192047119140625, 'stomach': 0.36012423038482666, 'black_buruma': 0.3422132134437561, 'official_alternate_costume': 0.2783987522125244, 'baseball': 0.38377970457077026, 'baseball_mitt': 0.32592540979385376, 'cropped_shirt': 0.35402947664260864, 'holding_baseball_bat': 0.2758416533470154, 'black_sports_bra': 0.3463800549507141, 'sports_bra': 0.28466159105300903, 'exercising': 0.2603980302810669, 'bike_jersey': 0.2661605477333069, 'patreon_username': 0.7087235450744629, 'patreon_logo': 0.560276210308075}
        >>> chars
        {'skadi_(arknights)': 0.5921452641487122}
        >>>
        >>> rating, features, chars = get_camie_tags('hutao.jpg')
        >>> rating
        {'general': 0.41121846437454224, 'sensitive': 0.4002530574798584, 'questionable': 0.03438958525657654, 'explicit': 0.04617959260940552}
        >>> features
        {'1girl': 0.8312125205993652, 'blush': 0.3996567726135254, 'cowboy_shot': 0.28660568594932556, 'long_hair': 0.7184156775474548, 'long_sleeves': 0.4706878066062927, 'looking_at_viewer': 0.5503140687942505, 'school_uniform': 0.365602970123291, 'shirt': 0.41183334589004517, 'sidelocks': 0.28638553619384766, 'smile': 0.3707748055458069, 'solo': 0.520854115486145, 'standing': 0.2960333526134491, 'tongue': 0.6556028127670288, 'tongue_out': 0.6966925859451294, 'very_long_hair': 0.5526134371757507, 'skirt': 0.6872812509536743, 'brown_hair': 0.5945607423782349, 'hair_ornament': 0.4464661478996277, 'hair_ribbon': 0.3646523952484131, 'outdoors': 0.37938451766967773, 'red_eyes': 0.5426545143127441, 'ribbon': 0.3027467727661133, 'bag': 0.8986430168151855, 'hair_between_eyes': 0.337802529335022, 'holding': 0.38589367270469666, 'pleated_skirt': 0.6475872993469238, 'school_bag': 0.666648805141449, 'ahoge': 0.4749193489551544, 'white_shirt': 0.27104783058166504, 'closed_mouth': 0.28101325035095215, 'collared_shirt': 0.37030768394470215, 'miniskirt': 0.32576680183410645, ':p': 0.4337637424468994, 'alternate_costume': 0.42441293597221375, 'black_skirt': 0.34694597125053406, 'twintails': 0.5711237192153931, 'open_clothes': 0.31017544865608215, 'nail_polish': 0.534726083278656, 'jacket': 0.4544385075569153, 'open_jacket': 0.27831193804740906, 'flower': 0.45064714550971985, 'plaid_clothes': 0.5494365096092224, 'plaid_skirt': 0.610480546951294, 'red_flower': 0.35928308963775635, 'contemporary': 0.37732189893722534, 'backpack': 0.5575172305107117, 'fingernails': 0.27776333689689636, 'cardigan': 0.3264558017253876, 'blue_jacket': 0.31882336735725403, 'ghost': 0.5534622073173523, 'red_nails': 0.38771501183509827, ':q': 0.3758758008480072, 'hair_flower': 0.39574217796325684, 'charm_(object)': 0.5394986271858215, 'handbag': 0.37014907598495483, 'black_bag': 0.44918346405029297, 'shoulder_bag': 0.5881174802780151, 'symbol-shaped_pupils': 0.5163478255271912, 'blue_cardigan': 0.28089386224746704, 'black_nails': 0.42480990290641785, 'bag_charm': 0.5010414123535156, 'plum_blossoms': 0.27618563175201416, 'flower-shaped_pupils': 0.5317837595939636}
        >>> chars
        {'hu_tao_(genshin_impact)': 0.8859397172927856, 'boo_tao_(genshin_impact)': 0.7348971366882324}
    """
    names = vnames(fmt)
    need_full = False
    for name in names:
        if '/' in name and name.split('/')[0] == 'initial':
            pass  # is initial
        else:
            need_full = True
            break

    image = load_image(image, force_background='white', mode='RGB')
    input_ = _get_camie_preprocessor(model_name)(image)[np.newaxis, ...]

    if need_full:
        model = _get_camie_model(model_name, is_full=True)
        init_embedding, init_logits, init_pred, refined_embedding, refined_logits, refined_pred = \
            model.run(["initial/embedding", "initial/logits", "initial/output", "embedding", "logits", "output"],
                      {'input': input_})
        init_values = _postprocess_embedding_values(
            pred=init_pred[0],
            logits=init_logits[0],
            embedding=init_embedding[0],
            model_name=model_name,
            mode=mode,
            thresholds=thresholds,
            drop_overlap=drop_overlap,
        )
        refined_values = _postprocess_embedding_values(
            pred=refined_pred[0],
            logits=refined_logits[0],
            embedding=refined_embedding[0],
            model_name=model_name,
            mode=mode,
            thresholds=thresholds,
            no_underline=no_underline,
            drop_overlap=drop_overlap,
        )
        values = {
            **refined_values,
            **{f'initial/{key}': value for key, value in init_values.items()},
            **{f'refined/{key}': value for key, value in refined_values.items()},
        }

    else:
        model = _get_camie_model(model_name, is_full=False)
        init_embedding, init_logits, init_pred = \
            model.run(["embedding", "logits", "output"], {'input': input_})
        init_values = _postprocess_embedding_values(
            pred=init_pred[0],
            logits=init_logits[0],
            embedding=init_embedding[0],
            model_name=model_name,
            mode=mode,
            thresholds=thresholds,
            no_underline=no_underline,
            drop_overlap=drop_overlap,
        )
        values = {
            **{f'initial/{key}': value for key, value in init_values.items()},
        }

    return vreplace(fmt, values)


@ts_lru_cache()
def _get_camie_emb_to_pred_model(model_name: str, is_refined: bool = False):
    """
    Load embedding-to-prediction conversion model.

    :param model_name: Model variant name
    :type model_name: str
    :param is_refined: Use refined embeddings (True) or initial embeddings (False)
    :type is_refined: bool
    :return: ONNX model session for embedding conversion
    :rtype: onnxruntime.InferenceSession
    """
    return open_onnx_model(hf_hub_download(
        repo_id=_REPO_ID,
        repo_type='model',
        filename=f'{model_name}/{"refined" if is_refined else "initial"}_emb_to_pred.onnx',
    ))


def convert_camie_emb_to_prediction(
        emb: np.ndarray,
        model_name: str = _DEFAULT_MODEL_NAME,
        is_refined: bool = True,
        mode: CamieModeTyping = 'balanced',
        thresholds: Optional[Union[float, Dict[str, float]]] = None,
        no_underline: bool = False,
        drop_overlap: bool = False,
        fmt: Any = ('rating', 'general', 'character'),
):
    """
    Convert stored embeddings back to tag predictions.

    Useful for reprocessing existing embeddings with new thresholds or formats.

    :param emb: Embedding vector(s) from previous inference
    :type emb: np.ndarray
    :param model_name: Original model variant name
    :type model_name: str
    :param is_refined: Whether embeddings come from refined stage, otherwise from initial stage
    :type is_refined: bool
    :param mode: Threshold selection strategy
    :type mode: CamieModeTyping
    :param thresholds: Custom threshold values
    :type thresholds: Optional[Union[float, Dict[str, float]]]
    :param no_underline: Remove underscores from tag names
    :type no_underline: bool
    :param drop_overlap: Remove overlapping tags in general category
    :type drop_overlap: bool
    :param fmt: Output format specification
    :type fmt: Any
    :return: Formatted results matching original prediction format
    :rtype: Any

    .. note::
        Modes for selection:

        - ``balanced``: Balanced precision/recall
        - ``high_precision``: Higher precision thresholds
        - ``high_recall``: Higher recall thresholds
        - ``micro_opt``: Micro-optimized thresholds
        - ``macro_opt``: Macro-optimized thresholds

    For batch processing (2-dim input), returns a list where each element corresponds
    to one embedding's predictions in the same format as single embedding output.

    Example:
        >>> import numpy as np
        >>> from imgutils.tagging import get_camie_tags, convert_camie_emb_to_prediction
        >>>
        >>> # extract the feature embedding, shape: (W, )
        >>> embedding = get_camie_tags('skadi.jpg', fmt='embedding')
        >>>
        >>> # convert to understandable result
        >>> rating, general, character = convert_camie_emb_to_prediction(embedding)
        >>> # these 3 dicts will be the same as that returned by `get_camie_tags('skadi.jpg')`
        >>>
        >>> # Batch processing, shape: (B, W)
        >>> embeddings = np.stack([
        ...     get_camie_tags('img1.jpg', fmt='embedding'),
        ...     get_camie_tags('img2.jpg', fmt='embedding'),
        ... ])
        >>> # results will be a list of (rating, general, character) tuples
        >>> results = convert_camie_emb_to_prediction(embeddings)
    """
    model = _get_camie_emb_to_pred_model(model_name=model_name, is_refined=is_refined)
    if len(emb.shape) == 1:
        logits, pred = model.run(["logits", "output"], {'embedding': emb[np.newaxis]})
        return vreplace(fmt, _postprocess_embedding_values(
            pred=pred[0],
            logits=logits[0],
            embedding=emb,
            model_name=model_name,
            mode=mode,
            thresholds=thresholds,
            no_underline=no_underline,
            drop_overlap=drop_overlap,
        ))
    else:
        retval = []
        for emb_item in emb:
            logits, pred = model.run(["logits", "output"], {'embedding': emb_item[np.newaxis]})
            retval.append(vreplace(fmt, _postprocess_embedding_values(
                pred=pred[0],
                logits=logits[0],
                embedding=emb_item,
                model_name=model_name,
                mode=mode,
                thresholds=thresholds,
                no_underline=no_underline,
                drop_overlap=drop_overlap,
            )))
        return retval
