from functools import lru_cache
from typing import Tuple, List

import numpy as np
import pandas as pd
from PIL import Image
from huggingface_hub import hf_hub_download

from ..data import ImageTyping, load_image
from ..utils import open_onnx_model


@lru_cache()
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


@lru_cache()
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
                          general_threshold: float = 0.5, character_threshold: float = 0.5):
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

    # Everything else is characters: pick anywhere prediction confidence > threshold
    character_names = [labels[i] for i in character_indexes]
    character_res = [x for x in character_names if x[1] > character_threshold]
    character_res = dict(character_res)

    return rating, general_res, character_res
