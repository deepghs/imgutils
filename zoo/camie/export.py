import copy
import datetime
import json
import os
import tempfile
from pprint import pformat

import onnx
import pandas as pd
import torch
from PIL import Image
from ditk import logging
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory
from hfutils.cache import delete_detached_cache
from hfutils.operate import get_hf_client, get_hf_fs, upload_directory_as_directory
from hfutils.repository import hf_hub_repo_file_url
from huggingface_hub import hf_hub_download
from natsort import natsorted
from safetensors.torch import save_model
from thop import profile, clever_format

from imgutils.data import load_image
from imgutils.preprocess import parse_pillow_transforms, create_torchvision_transforms, parse_torchvision_transforms
from imgutils.preprocess.pillow import PillowPadToSize, PillowToTensor, PillowCompose
from .model import create_initial_model, create_refined_model, InitialOnlyWrapper, FullWrapper
from .tags import load_tags
from ..utils import onnx_optimize

_P_TRANSFORM = parse_pillow_transforms(PillowCompose([
    PillowPadToSize((512, 512), interpolation=Image.LANCZOS, background_color=(0, 0, 0)),
    PillowToTensor(),
]))

_MODEL_MAP = {
    'initial': (_P_TRANSFORM, create_initial_model),
    'refined': (_P_TRANSFORM, create_refined_model),
}


def export_onnx_model(model, dummy_input, onnx_filename: str, is_full: bool = True,
                      opset_version: int = 17, verbose: bool = True, no_optimize: bool = False):
    if not is_full:
        wrapped_model = InitialOnlyWrapper(model)
    else:
        wrapped_model = FullWrapper(model)

    with torch.no_grad(), tempfile.TemporaryDirectory() as td:
        onnx_model_file = os.path.join(td, 'model.onnx')
        torch.onnx.export(
            wrapped_model,
            dummy_input,
            onnx_model_file,
            verbose=verbose,
            input_names=["input"],
            output_names=(
                ["embedding", "logits", "output"] if not is_full else
                ["initial/embedding", "initial/logits", "initial/output", "embedding", "logits", "output"]
            ),

            opset_version=opset_version,
            dynamic_axes={
                "input": {0: "batch"},
                "embedding": {0: "batch"},
                "logits": {0: "batch"},
                "output": {0: "batch"},
            } if not is_full else {
                "input": {0: "batch"},
                "initial/embedding": {0: "batch"},
                "initial/logits": {0: "batch"},
                "initial/output": {0: "batch"},
                "embedding": {0: "batch"},
                "logits": {0: "batch"},
                "output": {0: "batch"},
            }
        )

        model = onnx.load(onnx_model_file)
        if not no_optimize:
            model = onnx_optimize(model)

        output_model_dir, _ = os.path.split(onnx_filename)
        if output_model_dir:
            os.makedirs(output_model_dir, exist_ok=True)
        onnx.save(model, onnx_filename)


def get_threshold(model_name: str = 'initial'):
    with open(hf_hub_download(
            repo_id='Camais03/camie-tagger',
            repo_type='model',
            filename='model/thresholds.json',
    )) as f:
        return json.load(f)[model_name]


def extract(export_dir: str, model_name: str = "initial", no_optimize: bool = False):
    os.makedirs(export_dir, exist_ok=True)
    tp, model_fn = _MODEL_MAP[model_name]
    tprocess = create_torchvision_transforms(tp)
    model, created_at, (model_repo_id, model_filename) = model_fn()
    model = model.eval()

    sample_image = load_image(os.path.join('zoo', 'testfile', '6125785.jpg'), mode='RGB', force_background='white')
    dummy_input = tprocess(sample_image).unsqueeze(0)
    logging.info(f'Dummy input size: {dummy_input.shape!r}')

    logging.info('Trying to infer with dummy input ...')
    with torch.no_grad():
        dummy_init_embeddings, dummy_init_logits, dummy_refined_embeddings, dummy_refined_logits = model(dummy_input)
    logging.info(f'Shape of dummy init embedding: {dummy_init_embeddings.shape!r}')
    logging.info(f'Shape of dummy init logits: {dummy_init_logits.shape!r}')
    logging.info(f'Shape of dummy refined embedding: {dummy_refined_embeddings.shape!r}')
    logging.info(f'Shape of dummy refined logits: {dummy_refined_logits.shape!r}')

    threshold_file = os.path.join(export_dir, 'threshold.json')
    threshold_info = get_threshold(model_name)
    logging.info(f'Threshold of {model_name!r}:\n{pformat(threshold_info)}\n'
                 f'Saving thresholds to {threshold_file!r} ...')
    with open(threshold_file, 'w') as f:
        json.dump(threshold_info, f, sort_keys=True, ensure_ascii=False, indent=4)

    preprocess_file = os.path.join(export_dir, 'preprocess.json')
    logging.info(f'Saving preprocess configs to {preprocess_file!r} ...')
    with open(preprocess_file, 'w') as f:
        json.dump({
            'stages': parse_torchvision_transforms(tprocess)
        }, f, sort_keys=True, ensure_ascii=False, indent=4)

    logging.info('Profiling model ...')
    macs, params = profile(model, inputs=(dummy_input,))
    s_macs, s_params = clever_format([macs, params], "%.1f")
    logging.info(f'Params: {s_params}, FLOPs: {s_macs}')

    meta_file = os.path.join(export_dir, 'meta.json')
    meta_info = {
        'name': model_name,
        'input_size': dummy_input.shape[-1],
        'init_num_features': dummy_init_embeddings.shape[-1],
        'init_num_classes': dummy_init_logits.shape[-1],
        'refined_num_features': dummy_refined_embeddings.shape[-1],
        'refined_num_classes': dummy_refined_logits.shape[-1],
        **threshold_info['overall']['balanced'],
        'params': params,
        'flops': macs,
        'repo_id': model_repo_id,
        'model_filename': model_filename,
        'created_at': created_at,
    }
    with open(meta_file, 'w') as f:
        json.dump(meta_info, f, sort_keys=True, ensure_ascii=False, indent=4)

    model_weights_file = os.path.join(export_dir, 'model.safetensors')
    logging.info(f'Saving model weights to {model_weights_file!r} ...')
    save_model(
        model=model,
        filename=model_weights_file,
    )

    model_onnx_file = os.path.join(export_dir, 'model.onnx')
    logging.info(f'Exporting full model to {model_onnx_file!r} ...')
    export_onnx_model(
        model=model,
        dummy_input=dummy_input,
        onnx_filename=model_onnx_file,
        is_full=True,
        no_optimize=no_optimize,
    )

    initial_model_onnx_file = os.path.join(export_dir, 'model_initial_only.onnx')
    logging.info(f'Exporting initial-only model to {initial_model_onnx_file!r} ...')
    export_onnx_model(
        model=model,
        dummy_input=dummy_input,
        onnx_filename=initial_model_onnx_file,
        is_full=False,
        no_optimize=no_optimize,
    )

    df_tags = load_tags()
    tags_file = os.path.join(export_dir, 'selected_tags.csv')
    logging.info(f'Tags List:\n{df_tags}\n'
                 f'Saving to {tags_file!r} ...')
    df_tags.to_csv(tags_file, index=False)

    return meta_info


def sync(repository: str = 'deepghs/camie_tagger_onnx'):
    hf_client = get_hf_client()
    hf_fs = get_hf_fs()
    delete_detached_cache()
    if not hf_client.repo_exists(repo_id=repository, repo_type='model'):
        hf_client.create_repo(repo_id=repository, repo_type='model', private=True)
        attr_lines = hf_fs.read_text(f'{repository}/.gitattributes').splitlines(keepends=False)
        # attr_lines.append('*.json filter=lfs diff=lfs merge=lfs -text')
        # attr_lines.append('*.csv filter=lfs diff=lfs merge=lfs -text')
        hf_fs.write_text(f'{repository}/.gitattributes', os.linesep.join(attr_lines))

    if hf_client.file_exists(
            repo_id=repository,
            repo_type='model',
            filename='models.parquet',
    ):
        df_models = pd.read_parquet(hf_client.hf_hub_download(
            repo_id=repository,
            repo_type='model',
            filename='models.parquet',
        ))
        d_models = {item['name']: item for item in df_models.to_dict('records')}
    else:
        d_models = {}

    for model_name in ["initial", "refined"]:
        with TemporaryDirectory() as upload_dir:
            logging.info(f'Exporting model {model_name!r} ...')
            os.makedirs(os.path.join(upload_dir, model_name), exist_ok=True)
            try:
                extract(
                    export_dir=os.path.join(upload_dir, model_name),
                    model_name=model_name,
                    no_optimize=False,
                )
            except Exception:
                logging.exception(f'Error when exporting {model_name!r}, skipped.')
                continue

            with open(os.path.join(upload_dir, model_name, 'meta.json'), 'r') as f:
                meta_info = json.load(f)
            c_meta_info = copy.deepcopy(meta_info)
            d_models[meta_info['name']] = c_meta_info

            df_models = pd.DataFrame(list(d_models.values()))
            df_models = df_models.sort_values(by=['created_at'], ascending=False)
            df_models.to_parquet(os.path.join(upload_dir, 'models.parquet'), index=False)

            with open(os.path.join(upload_dir, 'README.md'), 'w') as f:
                print('---', file=f)
                print('pipeline_tag: image-classification', file=f)
                print('base_model:', file=f)
                for rid in natsorted(set(df_models['repo_id'][:100])):
                    print(f'- {rid}', file=f)
                print('language:', file=f)
                print('- en', file=f)
                print('tags:', file=f)
                print('- timm', file=f)
                print('- image', file=f)
                print('- dghs-imgutils', file=f)
                print('library_name: dghs-imgutils', file=f)
                print('license: gpl-3.0', file=f)
                print('---', file=f)
                print('', file=f)

                print('Camie Tagger ONNX Exported Version.', file=f)
                print('', file=f)

                print(f'# Models', file=f)
                print(f'', file=f)

                df_shown = pd.DataFrame([
                    {
                        "Name": f'[{item["name"]}]({hf_hub_repo_file_url(repo_id=item["repo_id"], repo_type="model", path=item["model_filename"])})',
                        'Params': clever_format(item["params"], "%.1f"),
                        'Flops': clever_format(item["flops"], "%.1f"),
                        'Input Size': item['input_size'],
                        "Features": item['refined_num_features'],
                        "Classes": item['refined_num_classes'],
                        'Threshold': f'{item["threshold"]:.4f}',
                        'F1 (MI/MA)': f'{item["micro_f1"]:.3f} / {item["macro_f1"]:.3f}',
                        'Precision (MI/MA)': f'{item["micro_precision"]:.3f} / {item["macro_precision"]:.3f}',
                        'Recall (MI/MA)': f'{item["micro_recall"]:.3f} / {item["macro_recall"]:.3f}',
                        'Created At': datetime.datetime.fromtimestamp(item['created_at']).strftime('%Y-%m-%d'),
                        'created_at': item['created_at'],
                    }
                    for item in df_models.to_dict('records')
                ])
                df_shown = df_shown.sort_values(by=['created_at'], ascending=[False])
                del df_shown['created_at']
                print(f'{plural_word(len(df_shown), "ONNX model")} exported in total.', file=f)
                print(f'', file=f)
                print(df_shown.to_markdown(index=False), file=f)
                print(f'', file=f)

            upload_directory_as_directory(
                repo_id=repository,
                repo_type='model',
                local_directory=upload_dir,
                path_in_repo='.',
                message=f'Export model {model_name!r}',
            )


if __name__ == '__main__':
    logging.try_init_root(level=logging.INFO)
    sync()
