import glob
import os.path
import random
import re
import tempfile
from functools import partial
from typing import Optional, Tuple

import click
import torch
from ditk import logging
from hbutils.testing import disable_output
from huggingface_hub import hf_hub_download
from sklearn import svm
from sklearn.metrics import accuracy_score
from torchvision import transforms
from tqdm.auto import tqdm

from imgutils.data import load_image
from .dataset import TEST_TRANSFORM
from .demo import _get_model_from_ckpt
from .model import CCIP
from .onnx import export_full_model_to_onnx, export_feat_model_to_onnx, export_metrics_model_to_onnx
from ..utils import GLOBAL_CONTEXT_SETTINGS
from ..utils import print_version as _origin_print_version

print_version = partial(_origin_print_version, 'zoo.ccip')


@click.group(context_settings={**GLOBAL_CONTEXT_SETTINGS})
@click.option('-v', '--version', is_flag=True,
              callback=print_version, expose_value=False, is_eager=True,
              help="Utils with pixiv resources.")
def cli():
    pass  # pragma: no cover


_CHECK_ITEMS = {
    'full': export_full_model_to_onnx,
    'feat': export_feat_model_to_onnx,
    'metrics': export_metrics_model_to_onnx,
}


def _sample_analysis(poss, negs, svm_samples: int = 10000):
    poss_cnt, negs_cnt = poss.shape[0], negs.shape[0]
    total = poss_cnt + negs_cnt
    if total > svm_samples:
        s_poss = poss[random.sample(range(poss_cnt), k=int(round(poss_cnt * svm_samples / total)))]
        s_negs = negs[random.sample(range(negs_cnt), k=int(round(negs_cnt * svm_samples / total)))]
    else:
        s_poss, s_negs = poss, negs

    s_poss, s_negs = s_poss.cpu(), s_negs.cpu()
    features = torch.cat([s_poss, s_negs]).detach().numpy()
    labels = torch.cat([torch.ones_like(s_poss), -torch.ones_like(s_negs)]).detach().numpy()

    model = svm.SVC(kernel='linear')  # 线性核
    model.fit(features.reshape(-1, 1), labels)
    predictions = model.predict(features.reshape(-1, 1))

    coef = model.coef_.reshape(-1)[0].tolist()
    inter = model.intercept_.reshape(-1)[0].tolist()
    threshold = -inter / coef

    return poss.mean().item(), poss.std().item(), \
        negs.mean().item(), negs.std().item(), \
        threshold, accuracy_score(labels, predictions)


@torch.no_grad()
def get_threshold_for_model(model: CCIP, preprocess, samples: int = 200) -> Tuple[float, float]:
    def _get_sim(x, y):
        x, y = load_image(x, mode='RGB'), load_image(y, mode='RGB')
        input_ = torch.stack([preprocess(x), preprocess(y)])
        return model(input_)[0][1]

    dataset_dir = 'test/testfile/dataset/images_xtiny_v0/'
    all_images = glob.glob(os.path.join(dataset_dir, '*', '*', '*.jpg'))
    all_chs = sorted(set([os.path.dirname(img) for img in all_images]))

    not_same_samples = []
    for _ in tqdm(range(samples)):
        x_ch, y_ch = random.sample(all_chs, k=2)
        x_img = random.choice(glob.glob(os.path.join(x_ch, '*.jpg')))
        y_img = random.choice(glob.glob(os.path.join(y_ch, '*.jpg')))
        not_same_samples.append(_get_sim(x_img, y_img))
    not_same_samples = torch.as_tensor(not_same_samples)

    same_samples = []
    for _ in tqdm(range(samples)):
        ch = random.choice(all_chs)
        x_img, y_img = random.sample(glob.glob(os.path.join(ch, '*.jpg')), k=2)
        same_samples.append(_get_sim(x_img, y_img))
    same_samples = torch.as_tensor(same_samples)

    _, _, _, _, threshold, accuracy = _sample_analysis(same_samples, not_same_samples, svm_samples=samples)
    return threshold, accuracy


@cli.command('onnx_check', help='Check onnx export is okay or not')
@click.option('--model', '-m', 'model', type=str, required=True,
              help='Model to be checked. ', show_default=True)
@click.option('--check', '-c', 'check_item', type=click.Choice(list(_CHECK_ITEMS.keys())), default=None,
              help='Model part to be checked. All parts will be checked when not given', show_default=True)
@click.option('--verbose', '-V', 'verbose', is_flag=True, type=bool, default=False,
              help='Show verbose information.', show_default=True)
@click.option('--output_dir', '-O', 'output_dir', type=click.Path(file_okay=False), default=None,
              help='Output directory of all models.', show_default=True)
@click.option('--threshold_samples', '-T', 'threshold_samples', type=int, default=500,
              help='Batch of samples to find threshold.', show_default=True)
def onnx_check(model: str, check_item: Optional[str] = None, verbose: bool = False,
               output_dir: Optional[str] = None, threshold_samples: int = 500):
    logging.try_init_root(logging.INFO)

    model, model_name = CCIP(model), model
    model.eval()
    if not check_item:
        check_items = list(_CHECK_ITEMS.keys())
    else:
        check_items = [check_item]

    logging.info('Finding threshold ...')
    threshold, accuracy = get_threshold_for_model(
        model,
        transforms.Compose(TEST_TRANSFORM + model.preprocess),
        samples=threshold_samples,
    )
    logging.info(f'Threshold: {threshold:.4f}, accuracy: {accuracy * 100.0:.2f}%')

    with tempfile.TemporaryDirectory() as td:
        for item in check_items:
            click.echo(click.style(f'Try exporting {model_name}-->{item} to onnx ... '), nl=False)
            onnx_filename = os.path.join(output_dir or td, re.sub(r'\W+', '-', f'{model_name}_{item}') + '.onnx')
            export_func = _CHECK_ITEMS[item]
            try:
                model = CCIP(model_name)  # necessary
                if verbose:
                    export_func(model, threshold, onnx_filename, verbose=verbose)
                else:
                    with disable_output():
                        export_func(model, threshold, onnx_filename, verbose=verbose)
            except:
                click.echo(click.style('FAILED', fg='red'), nl=True)
                raise
            else:
                click.echo(click.style('OK', fg='green'), nl=True)


MODELS = [
    ('caformer', 'ccip-caformer-2_fp32.ckpt'),
]


@cli.command('export', help='Export all models as onnx.',
             context_settings={**GLOBAL_CONTEXT_SETTINGS})
@click.option('--output_dir', '-O', 'output_dir', type=click.Path(file_okay=False), required=True,
              help='Output directory of all models.', show_default=True)
@click.option('--verbose', '-V', 'verbose', is_flag=True, type=bool, default=False,
              help='Show verbose information.', show_default=True)
@click.option('--threshold_samples', '-T', 'threshold_samples', type=int, default=500,
              help='Batch of samples to find threshold.', show_default=True)
def export(output_dir: str, verbose: bool = False, threshold_samples: int = 500):
    check_items = list(_CHECK_ITEMS.keys())

    for model_name, ckpt_name in MODELS:
        ckpt_file = hf_hub_download('deepghs/ccip', ckpt_name, repo_type='model')
        model, preprocess = _get_model_from_ckpt(model_name, ckpt_file, device='cpu', fp16=False)
        ckpt_body, _ = os.path.splitext(ckpt_name)

        logging.info(f'Finding threshold for {ckpt_name!r} ...')
        threshold, accuracy = get_threshold_for_model(
            model,
            transforms.Compose(TEST_TRANSFORM + model.preprocess),
            samples=threshold_samples,
        )
        logging.info(f'Threshold for {ckpt_file!r}: {threshold:.4f}, accuracy: {accuracy * 100.0:.2f}%')

        with tempfile.TemporaryDirectory() as td:
            for item in check_items:
                click.echo(click.style(f'Try exporting {ckpt_body!r}({model_name})-->{item} to onnx ... '), nl=False)
                onnx_filename = os.path.join(output_dir or td, f'{ckpt_body}_{item}.onnx')
                export_func = _CHECK_ITEMS[item]
                try:
                    model = CCIP(model_name)  # necessary
                    if verbose:
                        export_func(model, threshold, onnx_filename, verbose=verbose)
                    else:
                        with disable_output():
                            export_func(model, threshold, onnx_filename, verbose=verbose)
                except:
                    click.echo(click.style('FAILED', fg='red'), nl=True)
                    raise
                else:
                    click.echo(click.style('OK', fg='green'), nl=True)


if __name__ == '__main__':
    cli()
