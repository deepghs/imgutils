"""
Overview:
    Tagging utils based on deepdanbooru.

    .. warning::
        Due to the usage of an outdated model and training data in deepdanbooru,
        its performance is limited, and it is **not suitable for use as the main tagging model anymore**.
        The integration of this model within the present project serves only as a baseline for comparison,
        and it is advisable to avoid using this model extensively in practical applications.
"""
from typing import Tuple, List

import numpy as np
import pandas as pd
from PIL import Image
from huggingface_hub import hf_hub_download

from .overlap import drop_overlap_tags
from ..data import ImageTyping, load_image
from ..utils import open_onnx_model, ts_lru_cache


@ts_lru_cache()
def _get_deepdanbooru_labels():
    csv_file = hf_hub_download('deepghs/imgutils-models', 'deepdanbooru/deepdanbooru_tags.csv')
    df = pd.read_csv(csv_file)

    tag_names = df["name"].tolist()
    tag_real_names = df['real_name'].tolist()
    rating_indexes = list(np.where(df["category"] == 9)[0])
    general_indexes = list(np.where(df["category"] == 0)[0])
    character_indexes = list(np.where(df["category"] == 4)[0])
    return tag_names, tag_real_names, \
        rating_indexes, general_indexes, character_indexes


@ts_lru_cache()
def _get_deepdanbooru_model():
    return open_onnx_model(hf_hub_download(
        'deepghs/imgutils-models',
        'deepdanbooru/deepdanbooru.onnx',
    ))


def _image_preprocess(image: Image.Image) -> np.ndarray:
    o_width, o_height = image.size
    scale = 512.0 / max(o_width, o_height)
    f_width, f_height = map(lambda x: int(x * scale), (o_width, o_height))
    image = image.resize((f_width, f_height))

    data = np.asarray(image).astype(np.float32) / 255  # H x W x C
    height_pad_left = (512 - f_height) // 2
    height_pad_right = 512 - f_height - height_pad_left
    width_pad_left = (512 - f_width) // 2
    width_pad_right = 512 - f_width - width_pad_left
    data = np.pad(data, ((height_pad_left, height_pad_right), (width_pad_left, width_pad_right), (0, 0)),
                  mode='constant', constant_values=0.0)

    assert data.shape == (512, 512, 3), f'Shape (512, 512, 3) expected, but {data.shape!r} found.'
    return data.reshape((1, 512, 512, 3))  # B x H x W x C


def get_deepdanbooru_tags(image: ImageTyping, use_real_name: bool = False,
                          general_threshold: float = 0.5, character_threshold: float = 0.5,
                          drop_overlap: bool = False):
    """
    Overview:
        Get tags for anime image based on ``deepdanbooru`` model.

    :param image: Image to tagging.
    :param use_real_name: Use real name on danbooru. Due to the renaming and redirection of many tags
        on the Danbooru website after the training of ``deepdanbooru``,
        it may be necessary to use the latest tag names in some application scenarios.
        The default value of ``False`` indicates the use of the original tag names.
    :param general_threshold: Threshold for default tags, default is ``0.35``.
    :param character_threshold: Threshold for character tags, default is ``0.85``.
    :param drop_overlap: Drop overlap tags or not, default is ``False``.
    :return: Tagging results for levels, features and characters.

    Example:
        Here are some images for example

        .. image:: tagging_demo.plot.py.svg
           :align: center

        >>> from imgutils.tagging import get_deepdanbooru_tags
        >>>
        >>> rating, features, chars = get_deepdanbooru_tags('skadi.jpg')
        >>> rating
        {'rating:safe': 0.9897817373275757, 'rating:questionable': 0.010265946388244629, 'rating:explicit': 5.2809715270996094e-05}
        >>> features
        {'1girl': 0.9939777851104736, 'bangs': 0.5032387375831604, 'black_border': 0.9943548440933228, 'black_gloves': 0.5011609792709351, 'blue_sky': 0.6877802610397339, 'blush': 0.5543792843818665, 'breasts': 0.8268730640411377, 'cloud': 0.8504303693771362, 'cowboy_shot': 0.6008237600326538, 'crop_top': 0.6635787487030029, 'day': 0.8496965765953064, 'gloves': 0.6107005476951599, 'hair_between_eyes': 0.668294370174408, 'holding': 0.5619469285011292, 'holding_baseball_bat': 0.5141720771789551, 'letterboxed': 1.0, 'long_hair': 0.9884189963340759, 'looking_at_viewer': 0.5673105120658875, 'midriff': 0.6290556192398071, 'navel': 0.9631235003471375, 'no_hat': 0.7978747487068176, 'no_headwear': 0.7577926516532898, 'outdoors': 0.7118550539016724, 'parted_lips': 0.5452839136123657, 'pillarboxed': 0.9841411709785461, 'red_eyes': 0.958786129951477, 'shirt': 0.6720131039619446, 'short_sleeves': 0.7077711820602417, 'silver_hair': 0.6673924326896667, 'sky': 0.8709812760353088, 'solo': 0.9614333510398865, 'sportswear': 0.7786177396774292, 'standing': 0.6842771172523499, 'sweat': 0.9076308012008667, 'thighs': 0.580970823764801}
        >>> chars
        {'skadi_(arknights)': 0.9633345007896423}
        >>>
        >>> rating, features, chars = get_deepdanbooru_tags('hutao.jpg')
        >>> rating
        {'rating:safe': 0.9988503456115723, 'rating:questionable': 0.001651763916015625, 'rating:explicit': 0.00012505054473876953}
        >>> features
        {'1girl': 0.9829280972480774, ':p': 0.894218385219574, 'ahoge': 0.8733789920806885, 'backpack': 0.6322951316833496, 'bag': 0.9987058639526367, 'bag_charm': 0.9754379987716675, 'bangs': 0.6810564994812012, 'black_border': 0.9708781838417053, 'blush': 0.6356008052825928, 'bow': 0.5633733868598938, 'brick_wall': 0.5315935611724854, 'brown_hair': 0.9397273659706116, 'building': 0.9229896664619446, 'charm_(object)': 0.9006357789039612, 'city': 0.9020784497261047, 'cityscape': 0.9547432661056519, 'cowboy_shot': 0.5296419262886047, 'flower': 0.8253412246704102, 'hair_between_eyes': 0.5619839429855347, 'hair_flower': 0.8277763724327087, 'hair_ornament': 0.9356368780136108, 'hair_ribbon': 0.5288072824478149, 'jacket': 0.6336134076118469, 'letterboxed': 1.0, 'long_hair': 0.9703260064125061, 'looking_at_viewer': 0.8188960552215576, 'phone_screen': 0.9579574465751648, 'pillarboxed': 0.9954615235328674, 'plaid': 0.9725285172462463, 'plaid_skirt': 0.9638455510139465, 'pleated_skirt': 0.7226815819740295, 'red_eyes': 0.5321241021156311, 'red_nails': 0.5493080615997314, 'school_bag': 0.9863407611846924, 'school_uniform': 0.6794284582138062, 'shirt': 0.5062428116798401, 'shoulder_bag': 0.9325523972511292, 'skirt': 0.92237788438797, 'skyscraper': 0.7728171348571777, 'sleeves_past_wrists': 0.7257086038589478, 'smile': 0.5357837080955505, 'solo': 0.6939404010772705, 'thighhighs': 0.7054293155670166, 'tongue': 0.9990814924240112, 'tongue_out': 0.9992498755455017, 'twintails': 0.5012534260749817, 'very_long_hair': 0.7461410164833069}
        >>> chars
        {}
    """
    session = _get_deepdanbooru_model()
    _image_data = _image_preprocess(load_image(image, mode='RGB'))

    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    probs = session.run(output_names, {input_name: _image_data})[0]

    tag_names, tag_real_names, rating_indexes, general_indexes, character_indexes = _get_deepdanbooru_labels()
    labels: List[Tuple[str, float]] = list(zip(
        tag_real_names if use_real_name else tag_names,
        probs[0].astype(float).tolist(),
    ))

    # First 4 labels are actually ratings: pick one with argmax
    ratings_names = [labels[i] for i in rating_indexes]
    rating = dict(ratings_names)

    # Then we have general tags: pick anywhere prediction confidence > threshold
    general_names = [labels[i] for i in general_indexes]
    general_res = [x for x in general_names if x[1] > general_threshold]
    general_res = dict(general_res)
    if drop_overlap:
        general_res = drop_overlap_tags(general_res)

    # Everything else is characters: pick anywhere prediction confidence > threshold
    character_names = [labels[i] for i in character_indexes]
    character_res = [x for x in character_names if x[1] > character_threshold]
    character_res = dict(character_res)

    return rating, general_res, character_res
