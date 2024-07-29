import os.path
import re
from functools import lru_cache
from typing import List, Optional

import numpy as np
import onnx
import pandas as pd
from ditk import logging
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory
from hfutils.operate import upload_directory_as_directory, get_hf_fs
from huggingface_hub import hf_hub_download
from onnx.helper import make_tensor_value_info
from tqdm import tqdm

from imgutils.tagging.wd14 import MODEL_NAMES
from imgutils.utils import open_onnx_model
from .inv import _make_inverse
from .tags import _make_tag_info

logging.try_init_root(logging.INFO)


@lru_cache()
def _get_model_file(name) -> str:
    return hf_hub_download(
        repo_id=MODEL_NAMES[name],
        filename='model.onnx'
    )


@lru_cache()
def _get_model_tags_length(name) -> int:
    return len(pd.read_csv(hf_hub_download(
        repo_id=MODEL_NAMES[name],
        filename='selected_tags.csv',
    )))


def _seg_split(text):
    return tuple(filter(bool, re.split(r'[./]+', text)))


_FC_KEYWORDS_FOR_V2 = {'predictions_dense'}
_FC_NODE_PREFIXES_FOR_V3 = {
    "SwinV2": ('core_model', 'head', 'fc'),
    "SwinV2_v3": ('core_model', 'head', 'fc'),
    "ConvNext_v3": ('core_model', 'head', 'fc'),
    "ViT_v3": ('core_model', 'head'),
    "ViT_Large": ('core_model', 'head'),
    "EVA02_Large": ('core_model', 'head'),
}


def sync(tag_lazy_mode: bool = False, models: Optional[List[str]] = None):
    hf_fs = get_hf_fs()

    if models:
        _make_all = False
        _model_names = models
    else:
        _make_all = True
        _model_names = MODEL_NAMES

    import onnxruntime
    with TemporaryDirectory() as td:
        records = []
        for model_name in tqdm(_model_names):
            model_file = _get_model_file(model_name)
            logging.info(f'Model name: {model_name!r}, model file: {model_file!r}')
            logging.info(f'Loading model {model_name!r} ...')
            model = onnx.load(model_file)
            embs_outputs = []
            if model_name in _FC_NODE_PREFIXES_FOR_V3:
                prefix = _FC_NODE_PREFIXES_FOR_V3[model_name]

                def _is_fc(name):
                    return _seg_split(name)[:len(prefix)] == prefix
            else:
                def _is_fc(name):
                    return any(seg in _FC_KEYWORDS_FOR_V2 for seg in _seg_split(name))

            for node in model.graph.node:
                if _is_fc(node.name):
                    for input_name in node.input:
                        if not _is_fc(input_name):
                            logging.info(f'Input {input_name!r} for fc layer {node.name!r}.')
                            embs_outputs.append(input_name)

            logging.info(f'Embedding outputs: {embs_outputs!r}.')
            assert len(embs_outputs) == 1, f'Outputs: {embs_outputs!r}'
            # make_tensor_value_info(name=embs_outputs[0], elem_type=onnx.TensorProto.FLOAT, )
            model.graph.output.extend([onnx.ValueInfoProto(name=embs_outputs[0])])

            logging.info('Analysing via onnxruntime ...')
            session = onnxruntime.InferenceSession(model.SerializeToString())
            input_data = np.random.randn(1, 448, 448, 3).astype(np.float32)
            assert len(session.get_inputs()) == 1
            assert len(session.get_outputs()) == 2
            assert session.get_outputs()[1].name == embs_outputs[0]

            tags_data, embeddings = session.run([], {session.get_inputs()[0].name: input_data})
            logging.info(f'Tag output, shape: {tags_data.shape!r}, dtype: {tags_data.dtype!r}')
            logging.info(f'Embeddings output, shape: {embeddings.shape!r}, dtype: {embeddings.dtype!r}')
            assert tags_data.shape == (1, _get_model_tags_length(model_name))
            assert len(embeddings.shape) == 2 and embeddings.shape[0] == 1
            emb_width = embeddings.shape[-1]

            logging.info('Remaking model ...')
            model = onnx.load(model_file)
            model.graph.output.extend([make_tensor_value_info(
                name=embs_outputs[0],
                elem_type=onnx.TensorProto.FLOAT,
                shape=embeddings.shape,
            )])

            onnx_file = os.path.join(td, MODEL_NAMES[model_name], 'model.onnx')
            os.makedirs(os.path.dirname(onnx_file), exist_ok=True)
            onnx.save_model(model, onnx_file)

            logging.info(f'Loading and testing for the exported model {onnx_file!r}.')
            session = open_onnx_model(onnx_file)
            assert len(session.get_inputs()) == 1
            assert len(session.get_outputs()) == 2
            assert session.get_outputs()[1].name == embs_outputs[0]
            assert session.get_outputs()[1].shape == [1, emb_width]

            tags_data, embeddings = session.run([], {session.get_inputs()[0].name: input_data})
            logging.info(f'Tag output, shape: {tags_data.shape!r}, dtype: {tags_data.dtype!r}')
            logging.info(f'Embeddings output, shape: {embeddings.shape!r}, dtype: {embeddings.dtype!r}')
            assert tags_data.shape == (1, _get_model_tags_length(model_name))
            assert embeddings.shape == (1, emb_width)

            use_scale = 2000
            if hf_fs.exists(f'datasets/deepghs/wd14_tagger_inversion/{model_name}/samples_{use_scale}.npz'):
                _make_inverse(
                    model_name=model_name,
                    dst_dir=os.path.join(td, MODEL_NAMES[model_name]),
                    onnx_model_file=onnx_file,
                    scale=use_scale,
                )
                invertible = True
            else:
                invertible = False

            df = _make_tag_info(model_name, lazy_mode=tag_lazy_mode)
            assert len(df) == _get_model_tags_length(model_name)
            df.to_csv(os.path.join(td, MODEL_NAMES[model_name], 'tags_info.csv'), index=False)

            records.append({
                'Name': model_name,
                'Source Repository': f'[{MODEL_NAMES[model_name]}](https://huggingface.co/{MODEL_NAMES[model_name]})',
                'Tags Count': _get_model_tags_length(model_name),
                'Embedding Width': emb_width,
                'Inverse Supported': 'Yes' if invertible else 'No',
            })
            _get_model_file.cache_clear()
            _get_model_tags_length.cache_clear()

        df_records = pd.DataFrame(records)
        with open(os.path.join(td, 'README.md'), 'w') as f:
            print('---', file=f)
            print('license: apache-2.0', file=f)
            print('language:', file=f)
            print('- en', file=f)
            print('---', file=f)
            print('', file=f)

            print(
                f'This is onnx models based on [SmilingWolf](https://huggingface.co/SmilingWolf)\'s wd14 anime taggers, '
                f'which added the embeddings output as the second output.', file=f)
            print(f'', file=f)
            print(f'{plural_word(len(df_records), "model")} in total: ', file=f)
            print(f'', file=f)
            print(df_records.to_markdown(index=False), file=f)

        upload_directory_as_directory(
            repo_id='deepghs/wd14_tagger_with_embeddings',
            repo_type='model',
            local_directory=td,
            path_in_repo='.',
            message=f'Upload {plural_word(len(df_records), "models")}',
            clear=True if _make_all else False,
        )


if __name__ == '__main__':
    _MODELS = list(filter(bool, re.split('[,\s]+', os.environ.get('MODELS') or '')))
    sync(
        tag_lazy_mode=bool(os.environ.get('TAG_LAZY_MODE')),
        models=_MODELS if _MODELS else None,
    )
