import os.path
import re
import tempfile
from functools import partial
from typing import Optional

import click
from hbutils.testing import disable_output

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


@cli.command('onnx_check', help='Check onnx export is okay or not')
@click.option('--model', '-m', 'model', type=str, required=True,
              help='Model to be checked. ', show_default=True)
@click.option('--check', '-c', 'check_item', type=click.Choice(list(_CHECK_ITEMS.keys())), default=None,
              help='Model part to be checked. All parts will be checked when not given', show_default=True)
@click.option('--verbose', '-V', 'verbose', is_flag=True, type=bool, default=False,
              help='Show verbose information.', show_default=True)
@click.option('--output_dir', '-O', 'output_dir', type=click.Path(file_okay=False), default=None,
              help='Output directory of all models.', show_default=True)
def onnx_check(model: str, check_item: Optional[str] = None, verbose: bool = False,
               output_dir: Optional[str] = None):
    model, model_name = CCIP(model), model
    if not check_item:
        check_items = list(_CHECK_ITEMS.keys())
    else:
        check_items = [check_item]

    with tempfile.TemporaryDirectory() as td:
        for item in check_items:
            click.echo(click.style(f'Try exporting {model_name}-->{item} to onnx ... '), nl=False)
            onnx_filename = os.path.join(output_dir or td, re.sub(r'\W+', '-', f'{model_name}_{item}') + '.onnx')
            export_func = _CHECK_ITEMS[item]
            try:
                model = CCIP(model_name)  # necessary
                if verbose:
                    export_func(model, onnx_filename, verbose=verbose)
                else:
                    with disable_output():
                        export_func(model, onnx_filename, verbose=verbose)
            except:
                click.echo(click.style('FAILED', fg='red'), nl=True)
                raise
            else:
                click.echo(click.style('OK', fg='green'), nl=True)


if __name__ == '__main__':
    cli()
