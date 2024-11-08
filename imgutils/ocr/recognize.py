from typing import List, Tuple

import numpy as np
from huggingface_hub import hf_hub_download, HfFileSystem

from ..data import ImageTyping, load_image
from ..utils import open_onnx_model, ts_lru_cache

_HF_CLIENT = HfFileSystem()
_REPOSITORY = 'deepghs/paddleocr'


@ts_lru_cache()
def _open_ocr_recognition_model(model):
    return open_onnx_model(hf_hub_download(
        _REPOSITORY,
        f'rec/{model}/model.onnx',
    ))


@ts_lru_cache()
def _open_ocr_recognition_dictionary(model) -> List[str]:
    with open(hf_hub_download(
            _REPOSITORY,
            f'rec/{model}/dict.txt',
    ), 'r', encoding='utf-8') as f:
        dict_ = [line.strip() for line in f]

    return ['<blank>', *dict_, ' ']


def _text_decode(text_index, model: str, text_prob=None, is_remove_duplicate=False):
    retval = []
    ignored_tokens = [0]
    batch_size = len(text_index)
    for batch_idx in range(batch_size):
        selection = np.ones(len(text_index[batch_idx]), dtype=bool)
        if is_remove_duplicate:
            selection[1:] = text_index[batch_idx][1:] != text_index[batch_idx][:-1]
        for ignored_token in ignored_tokens:
            selection &= text_index[batch_idx] != ignored_token

        _dict = _open_ocr_recognition_dictionary(model)
        char_list = [_dict[text_id.item()] for text_id in text_index[batch_idx][selection]]
        if text_prob is not None:
            conf_list = text_prob[batch_idx][selection]
        else:
            conf_list = [1] * len(selection)
        if len(conf_list) == 0:
            conf_list = [0]

        text = ''.join(char_list)
        retval.append((text, np.mean(conf_list).tolist()))

    return retval


def _text_recognize(image: ImageTyping, model: str = 'ch_PP-OCRv4_rec',
                    is_remove_duplicate: bool = False) -> Tuple[str, float]:
    _ort_session = _open_ocr_recognition_model(model)
    expected_height = _ort_session.get_inputs()[0].shape[2]

    image = load_image(image, force_background='white', mode='RGB')
    r = expected_height / image.height
    new_height = int(round(image.height * r))
    new_width = int(round(image.width * r))
    image = image.resize((new_width, new_height))

    input_ = np.array(image).transpose((2, 0, 1)).astype(np.float32) / 255.0
    input_ = ((input_ - 0.5) / 0.5)[None, ...].astype(np.float32)
    _input_name = _ort_session.get_inputs()[0].name
    _output_name = _ort_session.get_outputs()[0].name
    output, = _ort_session.run([_output_name], {_input_name: input_})

    indices = output.argmax(axis=2)
    confs = output.max(axis=2)
    return _text_decode(indices, model, confs, is_remove_duplicate)[0]


@ts_lru_cache()
def _list_rec_models() -> List[str]:
    retval = []
    repo_segment_cnt = len(_REPOSITORY.split('/'))
    for item in _HF_CLIENT.glob(f'{_REPOSITORY}/rec/*/model.onnx'):
        retval.append(item.split('/')[repo_segment_cnt:][1])
    return retval
