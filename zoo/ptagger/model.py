import copy
import datetime
import json
import os.path

import numpy as np
import onnx
import onnxruntime
import pandas as pd
import torch
from PIL import Image
from ditk import logging
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory
from hfutils.cache import delete_detached_cache
from hfutils.operate import get_hf_fs, get_hf_client, upload_directory_as_directory
from hfutils.repository import hf_hub_repo_file_url
from natsort import natsorted
from procslib import get_model
from procslib.models.pixai_tagger import PixAITaggerInference
from thop import profile, clever_format
from timm.models._hub import save_for_hf
from torch import nn

from imgutils.preprocess import parse_torchvision_transforms
from test.testings import get_testfile
from zoo.utils import onnx_optimize
from zoo.wd14.tags import _get_tag_by_name


class ModuleWrapper(nn.Module):
    def __init__(self, base_module: nn.Module, classifier: nn.Module):
        super().__init__()
        self.base_module = base_module
        self.classifier = classifier

        self._output_features = None
        self._register_hook()

    def _register_hook(self):
        def hook_fn(module, input_tensor, output_tensor):
            assert isinstance(input_tensor, tuple) and len(input_tensor) == 1
            input_tensor = input_tensor[0]
            self._output_features = input_tensor

        self.classifier.register_forward_hook(hook_fn)

    def forward(self, x: torch.Tensor):
        logits = self.base_module(x)
        preds = torch.sigmoid(logits)

        if self._output_features is None:
            raise RuntimeError("Target module did not receive any input during forward pass")
        features, self._output_features = self._output_features, None
        assert all([x == 1 for x in features.shape[2:]]), f'Invalid feature shape: {features.shape!r}'
        features = torch.flatten(features, start_dim=1)

        return features, logits, preds


def load_model(model_name: str = "tagger_v_2_2_7"):
    hf_client = get_hf_client()
    try:
        logging.info(f'Try loading model {model_name!r} ...')
        model: PixAITaggerInference = get_model("pixai_tagger", model_version=model_name, device='cpu')
        created_at = hf_client.get_paths_info(
            repo_id=model.model_version_map[model_name]['repo_id'],
            repo_type='model',
            paths=[model.model_version_map[model_name]['ckpt_name']],
            expand=True
        )[0].last_commit.date.timestamp()
        model_repo_id = model.model_version_map[model_name]['repo_id']
        model_file = model.model_version_map[model_name]['ckpt_name']

    except (KeyError, ValueError):
        alt_model_name = "tagger_v_2_2_7"
        logging.info('Cannot directly load it, load from head weights ...')
        model: PixAITaggerInference = get_model("pixai_tagger", model_version=alt_model_name, device='cpu')
        state_dicts = torch.load(hf_client.hf_hub_download(
            repo_id=model.model_version_map[alt_model_name]['repo_id'],
            repo_type='model',
            filename=model.model_version_map[alt_model_name]['ckpt_name'],
        ), map_location="cpu")
        model_repo_id = model.model_version_map[alt_model_name]['repo_id']
        model_file = f'{model_name}.pth'
        state_dicts_head = torch.load(hf_client.hf_hub_download(
            repo_id=model.model_version_map[alt_model_name]['repo_id'],
            repo_type='model',
            filename=model_file,
        ), map_location="cpu")
        state_dicts['head.weight'] = state_dicts_head['head.0.weight']
        state_dicts['head.bias'] = state_dicts_head['head.0.bias']
        model.model.load_state_dict(state_dicts)
        model.model = model.model.to(model.device)
        model.model.eval()
        logging.info('Head weights loaded.')

        created_at = hf_client.get_paths_info(
            repo_id=model.model_version_map[alt_model_name]['repo_id'],
            repo_type='model',
            paths=[f'{model_name}.pth'],
            expand=True
        )[0].last_commit.date.timestamp()

    infer_model = model.model
    transforms = model.transform
    return model, infer_model, transforms, (model_repo_id, model_file), created_at


def extract(export_dir: str, model_name: str = "tagger_v_2_2_7", no_optimize: bool = False):
    hf_fs = get_hf_fs()
    hf_client = get_hf_client()

    os.makedirs(export_dir, exist_ok=True)

    raw_model, model, transforms, (model_repo_id, model_filename), created_at = load_model(model_name)
    raw_model: PixAITaggerInference
    image = Image.open(get_testfile('genshin_post.jpg'))
    dummy_input = transforms(image).unsqueeze(0)
    logging.info(f'Dummy input size: {dummy_input.shape!r}')

    with torch.no_grad():
        expected_dummy_output = model(dummy_input)
    logging.info(f'Dummy output size: {expected_dummy_output.shape!r}')

    classifier = model.get_classifier()
    classifier_position = None
    for name, module in model.named_modules():
        if module is classifier:
            classifier_position = name
            break
    if not classifier_position:
        raise RuntimeError(f'No classifier module found in model {type(model)}.')
    logging.info(f'Classifier module found at {classifier_position!r}:\n{classifier}')

    wrapped_model = ModuleWrapper(model, classifier=classifier)
    with torch.no_grad():
        conv_features, conv_output, conv_preds = wrapped_model(dummy_input)
    logging.info(f'Shape of embeddings: {conv_features.shape!r}')
    logging.info(f'Sample of expected logits:\n'
                 f'{expected_dummy_output[:, -10:]}\n'
                 f'Sample of actual logits:\n'
                 f'{conv_output[:, -10:]}')
    close_matrix = torch.isclose(expected_dummy_output, conv_output, atol=1e-3)
    ratio = close_matrix.type(torch.float32).mean()
    logging.info(f'{ratio * 100:.2f}% of the logits value are the same.')
    assert close_matrix.all(), 'Not all values can match.'

    matrix_data_file = os.path.join(export_dir, 'matrix.npz')
    bias = classifier.bias.detach().numpy()
    weight = classifier.weight.detach().numpy().T
    logging.info(f'Saving matrix data file to {matrix_data_file!r}, '
                 f'bias: {bias.dtype!r}{bias.shape!r}, weight: {weight.dtype!r}{weight.shape!r}.')
    np.savez(
        matrix_data_file,
        bias=bias,
        weight=weight,
    )
    expected_logits = conv_features.detach().numpy() @ weight + bias
    np.testing.assert_allclose(conv_output.detach().numpy(), expected_logits, rtol=1e-03, atol=1e-05)

    logging.info('Profiling model ...')
    macs, params = profile(model, inputs=(dummy_input,))
    s_macs, s_params = clever_format([macs, params], "%.1f")
    logging.info(f'Params: {s_params}, FLOPs: {s_macs}')

    logging.info('Exporting model weights ...')
    save_for_hf(
        model,
        expected_logits,
        safe_serialization='both',
    )

    with open(os.path.join(export_dir, 'meta.json'), 'w') as f:
        json.dump({
            'num_classes': conv_preds.shape[-1],
            'num_features': conv_features.shape[-1],
            'params': params,
            'flops': macs,
            'name': model_name,
            'model_cls': type(model).__name__,
            'input_size': dummy_input.shape[2],
            'repo_id': model_repo_id,
            'model_filename': model_filename,
            'created_at': created_at,
        }, f, indent=4, sort_keys=True)

    logging.info(f'Writing transforms:\n{transforms}')
    with open(os.path.join(export_dir, 'preprocess.json'), 'w') as f:
        json.dump({
            'stages': parse_torchvision_transforms(transforms),
        }, f, indent=4, sort_keys=True)

    df_p_tags = pd.read_csv(hf_client.hf_hub_download(
        repo_id='deepghs/site_tags',
        repo_type='dataset',
        filename='danbooru.donmai.us/tags.csv'
    ))
    logging.info(f'Loaded danbooru tags pool, columns: {df_p_tags.columns!r}')
    d_p_tags = {(item['category'], item['name']): item for item in df_p_tags.to_dict('records')}

    num_classes = raw_model.model_version_map[raw_model.model_version]['num_classes']
    logging.info(f'Num classes: {num_classes!r}')
    d_tags = {v: k for k, v in raw_model.tag_map.items()}
    r_tags = []
    for i in range(num_classes):
        category = 0 if i < raw_model.gen_tag_count else 4
        if (category, d_tags[i]) in d_p_tags:
            tag_id = d_p_tags[(category, d_tags[i])]['id']
            count = d_p_tags[(category, d_tags[i])]['post_count']
        else:
            logging.warning(f'Cannot find tag {d_tags[i]!r}, category: {category!r}.')
            tag_info = _get_tag_by_name(d_tags[i])
            if tag_info['name'] != d_tags[i]:
                logging.warning(f'Not found matching tags for {d_tags[i]!r}, will be ignored.')
                tag_id = -1
                count = -1
            else:
                logging.info(f'Tag info found from danbooru - {tag_info!r}.')
                tag_id = tag_info['id']
                count = tag_info['post_count']
        r_tags.append({
            'id': i,
            'tag_id': tag_id,
            'name': d_tags[i],
            'category': category,
            'count': count,
        })
    df_tags = pd.DataFrame(r_tags)
    tags_file = os.path.join(export_dir, 'selected_tags.csv')
    logging.info(f'Tags List:\n{df_tags}\n'
                 f'Saving to {tags_file!r} ...')
    df_tags.to_csv(tags_file, index=False)

    onnx_filename = os.path.join(export_dir, 'model.onnx')
    with TemporaryDirectory() as td:
        temp_model_onnx = os.path.join(td, 'model.onnx')
        logging.info(f'Exporting temporary ONNX model to {temp_model_onnx!r} ...')
        torch.onnx.export(
            wrapped_model,
            dummy_input,
            temp_model_onnx,
            input_names=['input'],
            output_names=['embedding', 'logits', 'output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'embedding': {0: 'batch_size'},
                'logits': {0: 'batch_size'},
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
        o_logits, o_embeddings = session.run(['logits', 'embedding'], {'input': dummy_input.numpy()})
        emb_1 = o_embeddings / np.linalg.norm(o_embeddings, axis=-1, keepdims=True)
        emb_2 = conv_features.numpy() / np.linalg.norm(conv_features.numpy(), axis=-1, keepdims=True)
        emb_sims = (emb_1 * emb_2).sum()
        logging.info(f'Similarity of the embeddings is {emb_sims:.5f}.')
        assert emb_sims >= 0.98, f'Similarity of the embeddings is {emb_sims:.5f}, ONNX validation failed.'


def sync(repository: str = 'onopix/pixai-tagger-onnx'):
    hf_client = get_hf_client()
    hf_fs = get_hf_fs()
    delete_detached_cache()
    if not hf_client.repo_exists(repo_id=repository, repo_type='model'):
        hf_client.create_repo(repo_id=repository, repo_type='model', private=True)
        attr_lines = hf_fs.read_text(f'{repository}/.gitattributes').splitlines(keepends=False)
        attr_lines.append('*.json filter=lfs diff=lfs merge=lfs -text')
        attr_lines.append('*.csv filter=lfs diff=lfs merge=lfs -text')
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

    for model_name in ["tagger_v_2_3_2", "tagger_v_2_2_7"]:
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
                print('---', file=f)
                print('', file=f)

                print('PixAI Tagger ONNX Exported Version.', file=f)
                print('', file=f)

                print(f'# Models', file=f)
                print(f'', file=f)

                df_shown = pd.DataFrame([
                    {
                        "Name": f'[{item["name"]}]({hf_hub_repo_file_url(repo_id=item["repo_id"], repo_type="model", path=item["model_filename"])})',
                        'Params': clever_format(item["params"], "%.1f"),
                        'Flops': clever_format(item["flops"], "%.1f"),
                        'Input Size': item['input_size'],
                        "Features": item['num_features'],
                        "Classes": item['num_classes'],
                        'Model': item['model_cls'],
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
    sync(
        repository='onopix/pixai-tagger-onnx'
    )
