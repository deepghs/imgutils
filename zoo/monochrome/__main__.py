import os.path
from functools import partial
from typing import List, Tuple

import click
import torch
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm

from .onnx import export_model_to_onnx
from .train_ import _KNOWN_MODELS
from ..utils import GLOBAL_CONTEXT_SETTINGS
from ..utils import print_version as _origin_print_version

print_version = partial(_origin_print_version, 'zoo.monochrome')


@click.group(context_settings={**GLOBAL_CONTEXT_SETTINGS})
@click.option('-v', '--version', is_flag=True,
              callback=print_version, expose_value=False, is_eager=True,
              help="Utils with pixiv resources.")
def cli():
    pass  # pragma: no cover


@cli.command('export_one', help='Export one model as onnx.',
             context_settings={**GLOBAL_CONTEXT_SETTINGS})
@click.option('--output', '-o', 'output', type=click.Path(dir_okay=False), required=True,
              help='Output path of feature model.', show_default=True)
@click.option('--feature_bins', '-b', 'feature_bins', type=int, default=256,
              help='Feature bins of input.', show_default=True)
@click.option('--ckpt', '-c', 'ckpt', type=click.Path(exists=True, dir_okay=False), required=True,
              help='Checkpoint file to export.', show_default=True)
@click.option('--model', '-m', 'model_name', type=click.Choice(list(_KNOWN_MODELS.keys())), required=True,
              help='Name of model to export.', show_default=True)
def export_one(output: str, feature_bins: int, ckpt: str, model_name: str):
    model = _KNOWN_MODELS[model_name]().float()
    model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    export_model_to_onnx(model, output, feature_bins=feature_bins)


_KNOWN_CKPTS: List[Tuple[str, str, int]] = [
    ('monochrome-alexnet_plus-320.ckpt', 'alexnet', 256),
    ('monochrome-alexnet_plus-500.ckpt', 'alexnet', 256),
]


@cli.command('export', help='Export all models as onnx.',
             context_settings={**GLOBAL_CONTEXT_SETTINGS})
@click.option('--output_dir', '-O', 'output_dir', type=click.Path(file_okay=False), required=True,
              help='Output directory of all models.', show_default=True)
def export(output_dir: str):
    for ckpt, model_name, feature_bins in tqdm(_KNOWN_CKPTS):
        model = _KNOWN_MODELS[model_name]().float()
        ckpt_file = hf_hub_download('deepghs/imgutils-models', f'monochrome/{ckpt}')
        model.load_state_dict(torch.load(ckpt_file, map_location='cpu'))
        filebody, _ = os.path.splitext(ckpt)
        output_file = os.path.join(output_dir, f'{filebody}.onnx')
        export_model_to_onnx(model, output_file, feature_bins=feature_bins)


if __name__ == '__main__':
    cli()
