from functools import lru_cache, partial
from typing import Optional

import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

from ._base import resize_image, cv2_resize, _get_image_edge
from ..data import ImageTyping, load_image
from ..utils import open_onnx_model


def _preprocess(input_image: Image.Image, detect_resolution: int = 512):
    input_image = np.array(input_image, dtype=np.uint8)
    input_image = resize_image(input_image, detect_resolution)
    return (input_image / 255.0).transpose(2, 0, 1)[None, ...].astype(np.float32)


@lru_cache()
def _open_la_model(coarse: bool):
    return open_onnx_model(hf_hub_download(
        'deepghs/imgutils-models',
        f'lineart/{"lineart.onnx" if not coarse else "lineart_coarse.onnx"}',
    ))


def get_edge_by_lineart(image: ImageTyping, coarse: bool = False, detect_resolution: int = 512):
    image = load_image(image, mode='RGB')
    output_, = _open_la_model(coarse).run(['output'], {'input': _preprocess(image, detect_resolution)})
    output_ = cv2_resize(output_[0].transpose(1, 2, 0), image.width, image.height)
    return 1.0 - output_.clip(0.0, 1.0)


def edge_image_with_lineart(image: ImageTyping, coarse: bool = False, detect_resolution: int = 512,
                            backcolor: str = 'white', forecolor: Optional[str] = None):
    return _get_image_edge(
        image,
        partial(get_edge_by_lineart, coarse=coarse, detect_resolution=detect_resolution),
        backcolor, forecolor
    )
