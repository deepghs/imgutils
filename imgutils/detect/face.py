from functools import lru_cache

from huggingface_hub import hf_hub_download

from ._yolo import _image_preprocess, _data_simple_postprocess
from ..data import ImageTyping, load_image, rgb_encode
from ..utils import open_onnx_model


@lru_cache()
def _open_face_detect_model(level: str = 's'):
    return open_onnx_model(hf_hub_download(
        'deepghs/imgutils-models',
        f'face_detect/face_detect_best_{level}.onnx'
    ))


def detect_faces(image: ImageTyping, level: str = 's', max_infer_size=1216,
                 conf_threshold: float = 0.25, iou_threshold: float = 0.7):
    image = load_image(image, mode='RGB')
    new_image, old_size, new_size = _image_preprocess(image, max_infer_size)

    data = rgb_encode(new_image)[None, ...]
    output, = _open_face_detect_model(level).run(['output0'], {'images': data})
    return _data_simple_postprocess(output[0], conf_threshold, iou_threshold, old_size, new_size)
