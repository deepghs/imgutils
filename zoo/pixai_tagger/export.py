import json
import os.path

import numpy as np
import onnx
import onnxruntime
import pandas as pd
import torch
from ditk import logging
from hbutils.system import TemporaryDirectory
from hfutils.operate import get_hf_client

from imgutils.data import load_image
from imgutils.preprocess import parse_torchvision_transforms
from zoo.pixai_tagger.tags import load_tags
from zoo.utils import onnx_optimize, get_testfile
from .min_script import EndpointHandler
from .onnx import get_model


def sync(src_repo: str, dst_repo: str, no_optimize: bool = False):
    hf_client = get_hf_client()
    if not hf_client.repo_exists(repo_id=dst_repo, repo_type='model'):
        hf_client.create_repo(repo_id=dst_repo, repo_type='model', private=True)

    handler = EndpointHandler(repo_id=src_repo)

    with TemporaryDirectory() as upload_dir:
        preprocessor = handler.transform
        preprocessor_file = os.path.join(upload_dir, 'preprocessor.json')
        logging.info(f'Dumping preprocessor:\n{preprocessor}\nto file {preprocessor_file!r}.')
        with open(preprocessor_file, 'w') as f:
            json.dump({
                'stages': parse_torchvision_transforms(handler.transform),
            }, f, sort_keys=True, ensure_ascii=False, indent=4)

        logging.info('Scanning tags ...')
        categories = np.zeros((len(handler.index_to_tag_map),), dtype=np.int32)
        idx = np.array(range(len(categories)))
        categories[idx < handler.gen_tag_count] = 0
        categories[idx >= handler.gen_tag_count] = 4
        df_src_tags = pd.DataFrame({
            'name': [v for _, v in sorted(handler.index_to_tag_map.items())],
            'category': categories,
        })
        df_tags = load_tags(df_src_tags)
        exts = []
        for titem in df_tags.to_dict('records'):
            if titem['category'] == 4:
                if titem['name'] in handler.character_ip_mapping:
                    exts.append(handler.character_ip_mapping[titem['name']])
                else:
                    exts.append([])
            else:
                exts.append([])
        df_tags['ips'] = exts
        logging.info(f'Tags:\n{df_tags}')
        df_tags.to_csv(os.path.join(upload_dir, 'selected_tags.csv'), index=False)

        with open(os.path.join(upload_dir, 'categories.json'), 'w') as f:
            json.dump([
                {
                    "category": 0,
                    "name": "general"
                },
                {
                    "category": 4,
                    "name": "character"
                },
            ], f, sort_keys=True, ensure_ascii=False, indent=4)
        df_th = pd.DataFrame([
            {'category': 0, 'name': 'general', 'threshold': handler.default_general_threshold},
            {'category': 4, 'name': 'character', 'threshold': handler.default_character_threshold},
        ])
        df_th.to_csv(os.path.join(upload_dir, 'thresholds.csv'), index=False)

        dummy_image = load_image(get_testfile('6125785.jpg'), mode='RGB', force_background='white')
        dummy_input = handler.transform(dummy_image).unsqueeze(0)
        wrapped_model, (conv_features, _) = get_model(handler.model, dummy_input)
        onnx_filename = os.path.join(upload_dir, 'model.onnx')
        with TemporaryDirectory() as td:
            temp_model_onnx = os.path.join(td, 'model.onnx')
            logging.info(f'Exporting temporary ONNX model to {temp_model_onnx!r} ...')
            torch.onnx.export(
                wrapped_model,
                dummy_input,
                temp_model_onnx,
                input_names=['input'],
                output_names=['embedding', 'output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'embedding': {0: 'batch_size'},
                    'output': {0: 'batch_size'},
                },
                opset_version=14,
                do_constant_folding=True,
                export_params=True,
                verbose=False,
                custom_opsets=None,
            )

            model = onnx.load(temp_model_onnx)
            if not no_optimize:
                logging.info('Optimizing onnx model ...')
                model = onnx_optimize(model)

            output_model_dir, _ = os.path.split(onnx_filename)
            if output_model_dir:
                os.makedirs(output_model_dir, exist_ok=True)
            logging.info(f'Complete model saving to {onnx_filename!r} ...')
            onnx.save(model, onnx_filename)

            session = onnxruntime.InferenceSession(onnx_filename)
            o_embeddings, = session.run(['embedding'], {'input': dummy_input.numpy()})
            emb_1 = o_embeddings / np.linalg.norm(o_embeddings, axis=-1, keepdims=True)
            emb_2 = conv_features.numpy() / np.linalg.norm(conv_features.numpy(), axis=-1, keepdims=True)
            emb_sims = (emb_1 * emb_2).sum()
            logging.info(f'Similarity of the embeddings is {emb_sims:.5f}.')
            assert emb_sims >= 0.98, f'Similarity of the embeddings is {emb_sims:.5f}, ONNX validation failed.'

        os.system(f'tree {upload_dir!r}')
        input()
    # print(df_tags)

    # pprint(handler.index_to_tag_map)
    # pprint(handler.gen_tag_count)
    # pprint(handler.character_tag_count)
    # pprint(handler.character_ip_mapping)


if __name__ == '__main__':
    logging.try_init_root(logging.INFO)
    sync(
        src_repo='pixai-labs/pixai-tagger-v0.9',
        dst_repo='deepghs/pixai-tagger-v0.9-onnx'
    )
