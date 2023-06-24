import datetime
import glob
import json
import os
import shutil
from functools import partial

import click
import torch
from PIL import Image
from ditk import logging
from hbutils.system import TemporaryDirectory
from hbutils.testing import disable_output
from huggingface_hub import hf_hub_download, HfApi, CommitOperationAdd
from natsort import natsorted
from torchvision import transforms
from tqdm.auto import tqdm

from test.testings import get_testfile
from zoo.ccip.demo import _get_model_from_ckpt
from .dataset import TEST_TRANSFORM
from .model import CCIP
from .onnx import get_scale_for_model, ModelWithScaleAlign, export_feat_model_to_onnx, export_metrics_model_to_onnx
from .plot import get_threshold_with_f1, plt_confusion_matrix, plt_export, plt_roc_curve, plt_p_curve, plt_r_curve, \
    plt_pr_curve, plt_f1_curve
from ..utils import GLOBAL_CONTEXT_SETTINGS
from ..utils import print_version as _origin_print_version

print_version = partial(_origin_print_version, 'zoo.ccip.publish')


@click.group(context_settings={**GLOBAL_CONTEXT_SETTINGS})
@click.option('-v', '--version', is_flag=True,
              callback=print_version, expose_value=False, is_eager=True,
              help="Utils with ccip publish.")
def cli():
    pass  # pragma: no cover


IMAGES_TEST_V1 = get_testfile('dataset', 'images_test_v1')


def _load_test_images():
    images = []
    cl_ids = []
    for img in natsorted(glob.glob(os.path.join(IMAGES_TEST_V1, '*', '*.jpg'))):
        images.append(img)
        cl_ids.append(int(os.path.basename(os.path.dirname(img))))

    return images, cl_ids


@torch.no_grad()
def _get_dist_matrix(model: CCIP, scale: float, batch: int = 32):
    preprocess = transforms.Compose(TEST_TRANSFORM + model.preprocess)
    images, cids = _load_test_images()

    inputs = torch.stack([preprocess(Image.open(item)) for item in tqdm(images, desc='Preprocessing')])

    feat_model = model.feature
    metrics_model = ModelWithScaleAlign(model.metrics, scale)
    if torch.cuda.is_available():
        inputs = inputs.cuda()
        feat_model = feat_model.cuda()
        metrics_model = metrics_model.cuda()

    feat_items = []
    for start in tqdm(range(0, len(images), batch), desc='Feature Extracting'):
        feat_items.append(feat_model(inputs[start: start + batch]))
    feats = torch.concat(feat_items)

    dist = metrics_model(feats).detach().cpu()
    cids = torch.tensor(cids)
    cmatrix = cids == cids.reshape(-1, 1)

    return dist, cmatrix


def create_plots(dist, cmatrix):
    # in plot function, score of pos samples should be greater than neg samples
    # so pos and neg should be reversed here!!!
    pos, neg = dist[~cmatrix].numpy(), dist[cmatrix].numpy()
    threshold, f1_score = get_threshold_with_f1(pos, neg)
    plots = {}

    y_true = cmatrix.reshape(-1).type(torch.int).numpy()
    y_pred = (dist <= threshold).reshape(-1).type(torch.int).numpy()
    accuracy = (y_true == y_pred).sum() / y_true.shape[0]
    logging.info(f'Threshold: {threshold:.4f}, f1 score: {f1_score:.4f}, accuracy: {accuracy * 100.0:.2f}%')

    logging.info('Creating confusion matrix ...')
    plots['confusion_matrix'] = plt_export(
        plt_confusion_matrix, y_true, y_pred,
        title=f'Confusion Matrix\nAccuracy: {accuracy * 100.0:.2f}%'
    )

    logging.info('Creating ROC Curve ...')
    plots['roc'] = plt_export(plt_roc_curve, pos, neg)

    logging.info('Creating Precision Curve ...')
    plots['precision'] = plt_export(plt_p_curve, pos, neg)

    logging.info('Creating Recall Curve ...')
    plots['recall'] = plt_export(plt_r_curve, pos, neg)

    logging.info('Creating PR Curve ...')
    plots['pr'] = plt_export(plt_pr_curve, pos, neg)

    logging.info('Creating F1 Curve ...')
    plots['f1'] = plt_export(plt_f1_curve, pos, neg)

    return (threshold, f1_score, accuracy), plots


def export_model_to_dir(file_in_repo: str, output_dir: str, repository: str = 'deepghs/ccip',
                        model_name: str = 'caformer', verbose: bool = False):
    logging.try_init_root(logging.INFO)

    os.makedirs(output_dir, exist_ok=True)
    ckpt_file = hf_hub_download(repository, file_in_repo, repo_type='model')
    model, preprocess = _get_model_from_ckpt(model_name, ckpt_file, device='cpu', fp16=False)

    model_ckpt_file = os.path.join(output_dir, 'model.ckpt')
    logging.info(f'Copying model file to {model_ckpt_file!r} ...')
    shutil.copyfile(ckpt_file, model_ckpt_file)

    scale = get_scale_for_model(model)
    dist, cmatrix = _get_dist_matrix(model, scale)
    (threshold, f1_score, accuracy), plots = create_plots(dist, cmatrix)
    metrics_file = os.path.join(output_dir, 'metrics.json')
    logging.info(f'Creating metric file {metrics_file!r} ...')
    with open(metrics_file, 'w') as f:
        json.dump({
            'threshold': threshold,
            'f1_score': f1_score,
            'accuracy': accuracy,
        }, fp=f, indent=4, sort_keys=True, ensure_ascii=False)

    for name, img in plots.items():
        plt_file = os.path.join(output_dir, f'plt_{name}.png')
        logging.info(f'Saving plotting file {plt_file!r} ...')
        img.save(plt_file)

    onnx_feat_file = os.path.join(output_dir, 'model_feat.onnx')
    logging.info(f'Creating ONNX feature model {onnx_feat_file!r} ...')
    model, preprocess = _get_model_from_ckpt(model_name, ckpt_file, device='cpu', fp16=False)
    with disable_output():
        export_feat_model_to_onnx(model, scale, onnx_feat_file, verbose=verbose)

    onnx_metrics_file = os.path.join(output_dir, 'model_metrics.onnx')
    logging.info(f'Creating ONNX metrics model {onnx_metrics_file!r} ...')
    model, preprocess = _get_model_from_ckpt(model_name, ckpt_file, device='cpu', fp16=False)
    with disable_output():
        export_metrics_model_to_onnx(model, scale, onnx_metrics_file, verbose=verbose)

    logging.info(f'Export complete, to {output_dir!r}.')


@cli.command('huggingface', help='Publish to huggingface')
@click.option('--model_repository', 'model_repository', type=str, default='deepghs/ccip',
              help='Source repository.', show_default=True)
@click.option('--file', '-f', 'model_file', type=str, required=True,
              help='Model file to export.', show_default=True)
@click.option('--model', '-m', 'model_name', type=str, default='caformer',
              help='Type of the model.', show_default=True)
@click.option('--name', '-n', 'name', type=str, default=None,
              help='Name of the model.', show_default=True)
@click.option('--repository', '-r', 'target_repository', type=str, default='deepghs/ccip_onnx',
              help='Target repository.', show_default=True)
@click.option('--verbose', '-V', 'verbose', is_flag=True, type=bool, default=False,
              help='Show verbose information.', show_default=True)
@click.option('--revision', '-R', 'revision', type=str, default='main',
              help='Revision for pushing the model.', show_default=True)
def huggingface(model_repository: str, model_file, model_name, name, target_repository, verbose, revision):
    hf_client = HfApi(token=os.environ.get('HF_TOKEN'))
    logging.info(f'Initialize repository {target_repository!r}')
    hf_client.create_repo(repo_id=target_repository, repo_type='model', exist_ok=True)

    name = name or os.path.splitext(os.path.basename(model_file))[0]
    with TemporaryDirectory() as td:
        export_model_to_dir(model_file, td, model_repository, model_name, verbose)

        current_time = datetime.datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')
        commit_message = f"Publish model {name}, on {current_time}"
        logging.info(f'Publishing model {name!r} to repository {target_repository!r} ...')
        hf_client.create_commit(
            target_repository,
            [
                CommitOperationAdd(
                    path_in_repo=f'{name}/{file}',
                    path_or_fileobj=os.path.join(td, file),
                ) for file in os.listdir(td)
            ],
            commit_message=commit_message,
            repo_type='model',
            revision=revision,
        )


if __name__ == '__main__':
    cli()
