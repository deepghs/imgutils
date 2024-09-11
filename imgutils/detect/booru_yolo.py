import ast
from functools import lru_cache
from pprint import pprint
from typing import Tuple, List

import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download

from test.testings import get_testfile
from ._yolo import _data_postprocess, _image_preprocess
from .visual import detection_visualize
from ..data import ImageTyping, load_image, rgb_encode
from ..utils import open_onnx_model


@lru_cache()
def _open_booru_yolo_model(model_name: str):
    return open_onnx_model(hf_hub_download(
        repo_id='deepghs/booru_yolo',
        repo_type='model',
        filename=f'{model_name}/model.onnx'
    ))


@lru_cache()
def _open_booru_yolo_labels(model_name: str):
    model = _open_booru_yolo_model(model_name)
    model_metadata = model.get_modelmeta()
    names_map = _safe_eval_names_str(model_metadata.custom_metadata_map['names'])
    labels = ['<Unknown>'] * (max(names_map.keys()) + 1)
    for id_, name in names_map.items():
        labels[id_] = name
    return labels


def _safe_eval_names_str(names_str):
    node = ast.parse(names_str, mode='eval')
    result = {}
    for key, value in zip(node.body.keys, node.body.values):
        if isinstance(key, (ast.Str, ast.Num)):
            key = ast.literal_eval(key)
        else:
            raise RuntimeError(f"Invalid key type: {key!r}, this should be a bug, "
                               f"please open an issue to dghs-imgutils.")  # pragma: no cover

        if isinstance(value, (ast.Str, ast.Num)):
            value = ast.literal_eval(value)
        else:
            raise RuntimeError(f"Invalid value type: {value!r}, this should be a bug, "
                               f"please open an issue to dghs-imgutils.")  # pragma: no cover

        result[key] = value

    return result


def detect_with_booru_yolo(image: ImageTyping, model_name: str = 'yolov8s_aa11',
                           max_infer_size: int = 640, conf_threshold: float = 0.25, iou_threshold: float = 0.7) \
        -> List[Tuple[Tuple[int, int, int, int], str, float]]:
    image = load_image(image, mode='RGB')
    model = _open_booru_yolo_model(model_name)
    labels = _open_booru_yolo_labels(model_name)
    new_image, old_size, new_size = _image_preprocess(image, max_infer_size)
    print(new_image)
    data = rgb_encode(new_image)[None, ...]
    output, = model.run(['output0'], {'images': data})
    return _data_postprocess(output[0], conf_threshold, iou_threshold, old_size, new_size, labels)


if __name__ == '__main__':
    # image = get_testfile('nude_girl.png')
    # image = get_testfile('complex_sex.jpg')
    # image = get_testfile('nian.png')
    image = get_testfile('nai3.png')
    d = detect_with_booru_yolo(image)
    pprint(d)
    plt.imshow(detection_visualize(image, d))
    plt.show()
