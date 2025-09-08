import json
import os.path

import numpy as np
import onnx
import onnxruntime
import pandas as pd
import torch
from ditk import logging
from hbutils.string import titleize
from hbutils.system import TemporaryDirectory
from hbutils.testing import vpip
from hfutils.operate import get_hf_client, upload_directory_as_directory
from hfutils.repository import hf_hub_repo_url, hf_hub_repo_file_url
from hfutils.utils import hf_normpath
from huggingface_hub import hf_hub_url
from thop import clever_format

from imgutils.data import load_image
from imgutils.preprocess import parse_torchvision_transforms
from zoo.pixai_tagger.tags import load_tags
from zoo.utils import onnx_optimize, get_testfile, torch_model_profile_via_calflops
from .min_script import EndpointHandler
from .onnx import get_model


def sync(src_repo: str, dst_repo: str, no_optimize: bool = False):
    hf_client = get_hf_client()
    if not hf_client.repo_exists(repo_id=dst_repo, repo_type='model'):
        hf_client.create_repo(repo_id=dst_repo, repo_type='model', private=True)

    handler = EndpointHandler(repo_id=src_repo)
    meta_info = {}
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

        d_category_names = {
            0: 'general',
            4: 'character',
        }
        with open(os.path.join(upload_dir, 'categories.json'), 'w') as f:
            json.dump([
                {
                    "category": cate_id,
                    "name": cate_name,
                } for cate_id, cate_name in sorted(d_category_names.items())
            ], f, sort_keys=True, ensure_ascii=False, indent=4)
        df_th = pd.DataFrame([
            {'category': 0, 'name': 'general', 'threshold': handler.default_general_threshold},
            {'category': 4, 'name': 'character', 'threshold': handler.default_character_threshold},
        ])
        df_th.to_csv(os.path.join(upload_dir, 'thresholds.csv'), index=False)

        dummy_image = load_image(get_testfile('6125785.jpg'), mode='RGB', force_background='white')
        dummy_input = handler.transform(dummy_image).unsqueeze(0).to(handler.device)
        flops, params, macs = torch_model_profile_via_calflops(model=handler.model, input_=dummy_input)
        meta_info['flops'] = flops
        meta_info['params'] = params
        meta_info['macs'] = macs
        new_meta_file = os.path.join(upload_dir, 'meta.json')
        logging.info(f'Saving metadata to {new_meta_file!r} ...')
        with open(new_meta_file, 'w') as f:
            json.dump(meta_info, f, indent=4, sort_keys=True, ensure_ascii=False)

        wrapped_model, (conv_features, _) = get_model(handler.model, dummy_input)
        conv_features = conv_features.detach().cpu()
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
            o_embeddings, = session.run(['embedding'], {'input': dummy_input.detach().cpu().numpy()})
            emb_1 = o_embeddings / np.linalg.norm(o_embeddings, axis=-1, keepdims=True)
            emb_2 = conv_features.numpy() / np.linalg.norm(conv_features.numpy(), axis=-1, keepdims=True)
            emb_sims = (emb_1 * emb_2).sum()
            logging.info(f'Similarity of the embeddings is {emb_sims:.5f}.')
            assert emb_sims >= 0.98, f'Similarity of the embeddings is {emb_sims:.5f}, ONNX validation failed.'

        with open(os.path.join(upload_dir, 'README.md'), 'w') as f:
            print('---', file=f)
            print('pipeline_tag: image-classification', file=f)
            print('base_model:', file=f)
            print(f'- {src_repo}', file=f)
            print('language:', file=f)
            print('- en', file=f)
            print('tags:', file=f)
            print('- image', file=f)
            print('- dghs-imgutils', file=f)
            print('library_name: dghs-imgutils', file=f)
            print('license: mit', file=f)
            print('---', file=f)
            print('', file=f)

            print(f'# ONNX Version for {src_repo}', file=f)
            print(f'', file=f)

            print(f'This is the ONNX-exported version of PixAI\'s tagger '
                  f'[{src_repo}]({hf_hub_repo_url(repo_id=src_repo, repo_type="model")}).', file=f)
            print(f'', file=f)

            s_flops, s_params, s_macs = clever_format([flops, params, macs], "%.1f")
            print(f'## Model Details', file=f)
            print(f'', file=f)
            print(f'- **Model Type:** Multilabel Image classification / feature backbone', file=f)
            print(f'- **Model Stats:**', file=f)
            print(f'  - Params: {s_params}', file=f)
            print(f'  - FLOPs / MACs: {s_flops} / {s_macs}', file=f)
            print(f'  - Image size: {dummy_input.shape[-1]} x {dummy_input.shape[-2]}', file=f)
            print(f'  - Tags Count: {len(df_tags)}', file=f)
            for category in sorted(set(df_tags['category'])):
                print(f'    - {titleize(d_category_names[category])} (#{category}) Tags Count: '
                      f'{len(df_tags[df_tags["category"] == category])}', file=f)
            print(f'', file=f)

            print(f'## How to Use', file=f)
            print(f'', file=f)

            imgutils_version = str(vpip('dghs-imgutils')._actual_version)
            sample_input = dummy_image
            if min(sample_input.width, sample_input.height) > 640:
                r = min(sample_input.width, sample_input.height) / 640
                new_width = int(sample_input.width / r)
                new_height = int(sample_input.height / r)
                sample_input = sample_input.resize((new_width, new_height))
            sample_input_file = os.path.join(upload_dir, 'sample.webp')
            sample_input_relfile = hf_normpath(os.path.relpath(sample_input_file, upload_dir))
            sample_input.save(sample_input_file)
            sample_input_url = hf_hub_url(repo_id=dst_repo, repo_type='model', filename=sample_input_relfile)
            sample_input_page_url = hf_hub_repo_file_url(repo_id=dst_repo, repo_type='model', path=sample_input_relfile)

            print(f'We provided a sample image for our code samples, '
                  f'you can find it [here]({sample_input_page_url}).', file=f)
            print(f'', file=f)

            print(f'Install [dghs-imgutils](https://github.com/deepghs/imgutils) with the following command', file=f)
            print(f'', file=f)
            print(f'```shell', file=f)
            print(f'pip install \'dghs-imgutils>={imgutils_version}\' torch huggingface_hub timm pillow pandas', file=f)
            print(f'```', file=f)
            print(f'', file=f)

        upload_directory_as_directory(
            repo_id=dst_repo,
            repo_type='model',
            local_directory=upload_dir,
            path_in_repo='.',
            message=f'Upload ONNX export of model {src_repo!r}'
        )

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
