import datetime
import glob
import json
import os
import shutil
from functools import partial
from typing import Tuple

import click
import numpy as np
import torch
from PIL import Image
from ditk import logging
from hbutils.system import TemporaryDirectory
from hbutils.testing import disable_output
from huggingface_hub import hf_hub_download, HfApi, CommitOperationAdd
from lighttuner.hpo import hpo, R, uniform, randint
from natsort import natsorted
from sklearn.cluster import OPTICS, DBSCAN
from sklearn.metrics import adjusted_rand_score, precision_score, recall_score, f1_score
from torchvision import transforms
from tqdm.auto import tqdm

try:
    from typing import Literal
except (ImportError, ModuleNotFoundError):
    from typing_extensions import Literal

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

    return dist, cids, cmatrix


def clustering_metrics(dist, cids, method: Literal['dbscan', 'optics'] = 'dbscan',
                       init_steps: int = 50, max_steps: int = 350,
                       min_samples_range: Tuple[int, int] = (2, 5)):
    assert method in {'dbscan', 'optics'}, f'Method {method!r} not found.'

    def _trans_id(x):
        max_id = 0
        _maps = {}
        retval = []
        for item in x:
            if item == -1:
                retval.append(max_id)
                max_id += 1
            else:
                if item not in _maps:
                    _maps[item] = max_id
                    max_id += 1
                retval.append(_maps[item])
        return retval

    @hpo
    def opt_func(v):  # this function is still usable after decorating
        min_samples, eps = v['min_samples'], v['eps']

        def _metric(x, y):
            return dist[int(x), int(y)].item()

        samples = np.array(range(cids.shape[0])).reshape(-1, 1)
        if method == 'dbscan':
            clustering = DBSCAN(eps=eps, min_samples=min_samples, metric=_metric).fit(samples)
        elif method == 'optics':
            clustering = OPTICS(max_eps=eps, min_samples=min_samples, metric=_metric).fit(samples)
        else:
            assert False, 'Should not reach here!'

        ret_ids = clustering.labels_.tolist()
        logging.info(f'Cluster result: {ret_ids!r}')
        return adjusted_rand_score(cids, _trans_id(ret_ids))

    logging.info('Waiting for HPO ...')
    params, score, _ = opt_func.bayes() \
        .init_steps(init_steps) \
        .max_steps(max_steps) \
        .maximize(R).max_workers(1).rank(10) \
        .spaces({
        'min_samples': randint(*min_samples_range),
        'eps': uniform(0.0, 0.5),
    }).run()

    return params, score


def create_plots(dist, cmatrix):
    # in plot function, score of pos samples should be greater than neg samples
    # so pos and neg should be reversed here!!!
    pos, neg = dist[~cmatrix].numpy(), dist[cmatrix].numpy()
    threshold, _ = get_threshold_with_f1(pos, neg)
    plots = {}

    y_true = cmatrix.reshape(-1).type(torch.int).numpy()
    y_pred = (dist <= threshold).reshape(-1).type(torch.int).numpy()
    f1 = f1_score(y_true, y_pred, pos_label=1)
    precision = precision_score(y_true, y_pred, pos_label=1)
    recall = recall_score(y_true, y_pred, pos_label=1)
    metrics = {
        'threshold': threshold,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
    }
    logging.info(f'Threshold: {threshold:.4f}, f1 score: {f1:.4f}, '
                 f'precision: {precision * 100.0:.2f}%, recall: {recall * 100.0:.2f}%')

    logging.info('Creating confusion matrix ...')
    plots['confusion_matrix_true'] = plt_export(
        plt_confusion_matrix, y_true, y_pred,
        normalize='true', title=f'Confusion Matrix (True)'
    )
    plots['confusion_matrix_pred'] = plt_export(
        plt_confusion_matrix, y_true, y_pred,
        normalize='pred', title=f'Confusion Matrix (Predict)'
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

    return threshold, metrics, plots


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
    dist, cids, cmatrix = _get_dist_matrix(model, scale)
    threshold, metrics, plots = create_plots(dist, cmatrix)
    metrics_file = os.path.join(output_dir, 'metrics.json')
    logging.info(f'Creating metric file {metrics_file!r} ...')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, fp=f, indent=4, sort_keys=True, ensure_ascii=False)

    clustering_file = os.path.join(output_dir, 'cluster.json')
    logging.info(f'Creating clustering measurement {clustering_file!r} ...')
    c_results = {}
    for cname, method, xrange in [
        ('dbscan_free', 'dbscan', (2, 5)),
        ('dbscan_2', 'dbscan', (2, 2)),
        ('optics', 'optics', (2, 5)),
    ]:
        params, score = clustering_metrics(dist, cids, method=method, min_samples_range=xrange)
        c_results[cname] = {**params, 'score': score}
    with open(clustering_file, 'w') as f:
        json.dump(c_results, fp=f, indent=4, sort_keys=True, ensure_ascii=False)

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
