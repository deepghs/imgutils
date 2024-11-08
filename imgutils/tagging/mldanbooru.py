"""
Overview:
    Tagging utils based on ML-danbooru which is provided by 7eu7d7. The code is here:
    `7eu7d7/ML-Danbooru <https://github.com/7eu7d7/ML-Danbooru>`_ .
"""
from typing import Tuple, List

import numpy as np
import pandas as pd
from PIL import Image
from huggingface_hub import hf_hub_download

from .overlap import drop_overlap_tags
from ..data import load_image, ImageTyping
from ..utils import open_onnx_model, ts_lru_cache


@ts_lru_cache()
def _open_mldanbooru_model():
    return open_onnx_model(hf_hub_download('deepghs/ml-danbooru-onnx', 'ml_caformer_m36_dec-5-97527.onnx'))


def _resize_align(image: Image.Image, size: int, keep_ratio: float = True, align: int = 4) -> Image.Image:
    if not keep_ratio:
        target_size = (size, size)
    else:
        min_edge = min(image.size)
        target_size = (
            int(image.size[0] / min_edge * size),
            int(image.size[1] / min_edge * size),
        )

    target_size = (
        (target_size[0] // align) * align,
        (target_size[1] // align) * align,
    )

    return image.resize(target_size, resample=Image.BILINEAR)


def _to_tensor(image: Image.Image):
    # noinspection PyTypeChecker
    img: np.ndarray = np.array(image, dtype=np.uint8, copy=True)
    img = img.reshape((image.size[1], image.size[0], len(image.getbands())))

    # put it from HWC to CHW format
    img = img.transpose((2, 0, 1))
    return img.astype(np.float32) / 255


@ts_lru_cache()
def _get_mldanbooru_labels(use_real_name: bool = False) -> Tuple[List[str], List[int], List[int]]:
    path = hf_hub_download('deepghs/imgutils-models', 'mldanbooru/mldanbooru_tags.csv')
    df = pd.read_csv(path)

    return df["name"].tolist() if not use_real_name else df['real_name'].tolist()


def get_mldanbooru_tags(image: ImageTyping, use_real_name: bool = False,
                        threshold: float = 0.7, size: int = 448, keep_ratio: bool = False,
                        drop_overlap: bool = False):
    """
    Overview:
        Tagging image with ML-Danbooru, similar to
        `deepghs/ml-danbooru-demo <https://huggingface.co/spaces/deepghs/ml-danbooru-demo>`_.

    :param image: Image to tagging.
    :param use_real_name: Use real name on danbooru. Due to the renaming and redirection of many tags
        on the Danbooru website after the training of ``deepdanbooru``,
        it may be necessary to use the latest tag names in some application scenarios.
        The default value of ``False`` indicates the use of the original tag names.
    :param threshold: Threshold for tags, default is ``0.7``.
    :param size: Size when passing the resized image into model, default is ``448``.
    :param keep_ratio: Keep the original ratio between height and width when passing the image into
        model, default is ``False``.
    :param drop_overlap: Drop overlap tags or not, default is ``False``.

    Example:
        Here are some images for example

        .. image:: tagging_demo.plot.py.svg
           :align: center

        >>> import os
        >>> from imgutils.tagging import get_mldanbooru_tags
        >>>
        >>> get_mldanbooru_tags('skadi.jpg')
        {'1girl': 0.9999984502792358, 'long_hair': 0.9999946355819702, 'red_eyes': 0.9994951486587524, 'navel': 0.998144268989563, 'breasts': 0.9978417158126831, 'solo': 0.9941409230232239, 'shorts': 0.9799384474754333, 'gloves': 0.979142427444458, 'very_long_hair': 0.961823582649231, 'looking_at_viewer': 0.961323618888855, 'silver_hair': 0.9490893483161926, 'large_breasts': 0.9450850486755371, 'midriff': 0.9425153136253357, 'sweat': 0.9409335255622864, 'thighs': 0.9319437146186829, 'crop_top': 0.9265308976173401, 'baseball_bat': 0.9259042143821716, 'sky': 0.922250509262085, 'holding': 0.9199565052986145, 'outdoors': 0.9175475835800171, 'day': 0.9102761745452881, 'black_gloves': 0.9076938629150391, 'stomach': 0.9052775502204895, 'shirt': 0.8938589692115784, 'cowboy_shot': 0.8894285559654236, 'bangs': 0.8891903162002563, 'blue_sky': 0.8845980763435364, 'parted_lips': 0.8842408061027527, 'hair_between_eyes': 0.8659475445747375, 'sportswear': 0.862621009349823, 'no_headwear': 0.8616052865982056, 'cloud': 0.8562789559364319, 'short_shorts': 0.8555729389190674, 'no_hat': 0.8533340096473694, 'black_shorts': 0.8477485775947571, 'short_sleeves': 0.8430152535438538, 'low-tied_long_hair': 0.8340626955032349, 'crop_top_overhang': 0.8266023397445679, 'holding_baseball_bat': 0.8222048282623291, 'standing': 0.8202669620513916, 'black_shirt': 0.8061150312423706, 'ass_visible_through_thighs': 0.7803354859352112, 'thigh_gap': 0.7789446711540222, 'arms_up': 0.7052110433578491}
        >>>
        >>> get_mldanbooru_tags('hutao.jpg')
        {'1girl': 0.9999866485595703, 'skirt': 0.997043788433075, 'tongue': 0.9969649910926819, 'hair_ornament': 0.9957101345062256, 'tongue_out': 0.9928386807441711, 'flower': 0.9886980056762695, 'twintails': 0.9864778518676758, 'ghost': 0.9769423007965088, 'hair_flower': 0.9747489094734192, 'bag': 0.9736957550048828, 'long_hair': 0.9388670325279236, 'backpack': 0.9356311559677124, 'brown_hair': 0.91000896692276, 'cardigan': 0.8955123424530029, 'red_eyes': 0.8910233378410339, 'plaid': 0.8904104828834534, 'looking_at_viewer': 0.8881211280822754, 'school_uniform': 0.8876776695251465, 'outdoors': 0.8864808678627014, 'jacket': 0.8810517191886902, 'plaid_skirt': 0.8798807263374329, 'ahoge': 0.8765745162963867, 'pleated_skirt': 0.8737136125564575, 'nail_polish': 0.8650439381599426, 'solo': 0.8613706827163696, 'blue_cardigan': 0.8571277260780334, 'bangs': 0.8333670496940613, 'very_long_hair': 0.8160212635993958, 'eyebrows_visible_through_hair': 0.8122442364692688, 'hairclip': 0.8091571927070618, 'red_nails': 0.8082079887390137, ':p': 0.8048468232154846, 'long_sleeves': 0.8042327165603638, 'shirt': 0.7984272241592407, 'blazer': 0.794708251953125, 'ribbon': 0.78981614112854, 'hair_ribbon': 0.7892146110534668, 'star-shaped_pupils': 0.7867060899734497, 'gradient_hair': 0.786359965801239, 'white_shirt': 0.7790888547897339, 'brown_skirt': 0.7760675549507141, 'symbol-shaped_pupils': 0.774523913860321, 'smile': 0.7721588015556335, 'hair_between_eyes': 0.7697228789329529, 'cowboy_shot': 0.755959689617157, 'multicolored_hair': 0.7477189898490906, 'blush': 0.7476690411567688, 'railing': 0.7476617693901062, 'blue_jacket': 0.7458406090736389, 'sleeves_past_wrists': 0.741143524646759, 'day': 0.7364678978919983, 'collared_shirt': 0.7193643450737, 'red_neckwear': 0.7108616828918457, 'flower-shaped_pupils': 0.7086325287818909, 'miniskirt': 0.7055293321609497, 'holding': 0.7039415836334229, 'open_clothes': 0.7018357515335083}

    .. note::
        ML-Danbooru only contains generic tags, so the return value will not be splitted like that in
        :func:`imgutils.tagging.deepdanbooru.get_deepdanbooru_tags` or
        :func:`imgutils.tagging.wd14.get_wd14_tags`.
    """
    image = load_image(image, mode='RGB')
    real_input = _to_tensor(_resize_align(image, size, keep_ratio))
    real_input = real_input.reshape(1, *real_input.shape)

    model = _open_mldanbooru_model()
    native_output, = model.run(['output'], {'input': real_input})

    output = (1 / (1 + np.exp(-native_output))).reshape(-1)
    tags = _get_mldanbooru_labels(use_real_name)
    pairs = sorted([(tags[i], ratio) for i, ratio in enumerate(output)], key=lambda x: (-x[1], x[0]))

    general_tags = {tag: float(ratio) for tag, ratio in pairs if ratio >= threshold}
    if drop_overlap:
        general_tags = drop_overlap_tags(general_tags)
    return general_tags
