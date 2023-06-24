import os.path
import tempfile
from functools import partial
from typing import Optional

import click
from ditk import logging
from hbutils.testing import disable_output
from huggingface_hub import hf_hub_download

from .demo import _get_model_from_ckpt
from .model import CCIP
from .onnx import export_feat_model_to_onnx, export_metrics_model_to_onnx, get_scale_for_model, \
    export_full_model_to_onnx
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


@cli.command('onnx_check', help='Check onnx export is okay or not')
@click.option('--model', '-m', 'model', type=str, required=True,
              help='Model to be checked. ', show_default=True)
@click.option('--verbose', '-V', 'verbose', is_flag=True, type=bool, default=False,
              help='Show verbose information.', show_default=True)
@click.option('--output_dir', '-O', 'output_dir', type=click.Path(file_okay=False), default=None,
              help='Output directory of all models.', show_default=True)
def onnx_check(model: str, verbose: bool = False, output_dir: Optional[str] = None):
    logging.try_init_root(logging.INFO)

    model, model_name = CCIP(model), model
    model.eval()

    logging.info('Finding threshold ...')
    scale = get_scale_for_model(model)
    logging.info(f'Scale: {scale:.4f}')

    with tempfile.TemporaryDirectory() as td:
        for item in ['feat', 'metrics']:
            click.echo(click.style(f'Try exporting {model_name}-->{item} to onnx ... '), nl=False)
            onnx_filename = os.path.join(output_dir or td, f'{model_name}_{item}.onnx')
            export_func = _CHECK_ITEMS[item]
            try:
                model = CCIP(model_name)  # necessary
                if verbose:
                    export_func(model, scale, onnx_filename, verbose=verbose)
                else:
                    with disable_output():
                        export_func(model, scale, onnx_filename, verbose=verbose)
            except:
                click.echo(click.style('FAILED', fg='red'), nl=True)
                raise
            else:
                click.echo(click.style('OK', fg='green'), nl=True)


MODELS = [
    # ('caformer', 'ccip-caformer-2_fp32.ckpt'),
    # ('caformer', 'ccip-caformer-4_fp32.ckpt'),
    # ('caformer', 'ccip-caformer-5_fp32.ckpt'),
    # ('caformer', 'ccip-caformer-23_randaug_fp32.ckpt'),
    ('caformer', 'ccip-caformer-24-randaug-pruned.ckpt'),
    ('caformer_query', 'ccip-caformer_query-12.ckpt'),
]


@cli.command('export', help='Export all models as onnx.',
             context_settings={**GLOBAL_CONTEXT_SETTINGS})
@click.option('--repository', '-r', 'repository', type=str, default='deepghs/ccip',
              help='Source repository.', show_default=True)
@click.option('--output_dir', '-O', 'output_dir', type=click.Path(file_okay=False), required=True,
              help='Output directory of all models.', show_default=True)
@click.option('--verbose', '-V', 'verbose', is_flag=True, type=bool, default=False,
              help='Show verbose information.', show_default=True)
def export(repository: str, output_dir: str, verbose: bool = False):
    for model_name, ckpt_name in MODELS:
        ckpt_file = hf_hub_download(repository, ckpt_name, repo_type='model')
        model, preprocess = _get_model_from_ckpt(model_name, ckpt_file, device='cpu', fp16=False)
        ckpt_body, _ = os.path.splitext(ckpt_name)

        scale = get_scale_for_model(model)
        logging.info(f'Scale for {ckpt_file!r}: {scale:.4f}')

        with tempfile.TemporaryDirectory() as td:
            for item in ['feat', 'metrics']:
                click.echo(click.style(f'Try exporting {ckpt_body!r}({model_name})'
                                       f'-->{item} to onnx ... '), nl=False)
                onnx_filename = os.path.join(output_dir or td, f'{ckpt_body}_{item}.onnx')
                export_func = _CHECK_ITEMS[item]
                try:
                    model, preprocess = _get_model_from_ckpt(model_name, ckpt_file, device='cpu', fp16=False)
                    if verbose:
                        export_func(model, scale, onnx_filename, verbose=verbose)
                    else:
                        with disable_output():
                            export_func(model, scale, onnx_filename, verbose=verbose)
                except:
                    click.echo(click.style('FAILED', fg='red'), nl=True)
                    raise
                else:
                    click.echo(click.style('OK', fg='green'), nl=True)


if __name__ == '__main__':
    cli()
