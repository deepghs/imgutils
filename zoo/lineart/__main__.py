import os.path
import re
import tempfile
from functools import partial
from typing import Optional

import click
from hbutils.testing import disable_output
from tqdm.auto import tqdm

from .hed import _MyHEDDetector
from .la import _MyLineartDetector
from .la_anime import _MyLineartAnimeDetector
from .pidi import _MyPidiNetDetector
from ..utils import GLOBAL_CONTEXT_SETTINGS
from ..utils import print_version as _origin_print_version

_ONNX_ITEMS = {
    'lineart': (_MyLineartDetector, {'coarse': False}),
    'lineart_coarse': (_MyLineartDetector, {'coarse': True}),
    'lineart_anime': (_MyLineartAnimeDetector, {}),
    'pidi': (_MyPidiNetDetector, {}),
    'hed': (_MyHEDDetector, {}),
}


def _get_detector(name):
    _cls, _kwargs = _ONNX_ITEMS[name]
    return _cls.from_pretrained("lllyasviel/Annotators"), _kwargs


print_version = partial(_origin_print_version, 'zoo.lineart')


@click.group(context_settings={**GLOBAL_CONTEXT_SETTINGS})
@click.option('-v', '--version', is_flag=True,
              callback=print_version, expose_value=False, is_eager=True,
              help="Utils with pixiv resources.")
def cli():
    pass  # pragma: no cover


@cli.command('onnx_check', help='Check onnx export is okay or not')
@click.option('--model', '-m', 'model', type=str, required=None,
              help='Model to be checked. Check all when not given', show_default=True)
@click.option('--verbose', '-V', 'verbose', is_flag=True, type=bool, default=False,
              help='Show verbose information.', show_default=True)
@click.option('--output_dir', '-O', 'output_dir', type=click.Path(file_okay=False), default=None,
              help='Output directory of all models.', show_default=True)
def onnx_check(model: Optional[str] = None, verbose: bool = False,
               output_dir: Optional[str] = None):
    if model is None:
        models = list(_ONNX_ITEMS.keys())
    else:
        models = [model]

    with tempfile.TemporaryDirectory() as td:
        for item in models:
            click.echo(click.style(f'Try exporting {item} to onnx ...'), nl=False)
            onnx_filename = os.path.join(output_dir or td, re.sub(r'\W+', '-', f'{item}') + '.onnx')
            try:
                _model, _kwargs = _get_detector(item)
                if verbose:
                    _model.export_onnx(onnx_filename, **_kwargs, verbose=verbose)
                else:
                    with disable_output():
                        _model.export_onnx(onnx_filename, **_kwargs, verbose=verbose)
            except:
                click.echo(click.style('FAILED', fg='red'), nl=True)
                raise
            else:
                click.echo(click.style('OK', fg='green'), nl=True)


@cli.command('export', help='Export all models as onnx.',
             context_settings={**GLOBAL_CONTEXT_SETTINGS})
@click.option('--output_dir', '-O', 'output_dir', type=click.Path(file_okay=False), required=True,
              help='Output directory of all models.', show_default=True)
def export(output_dir: str):
    for key in tqdm(_ONNX_ITEMS.keys()):
        _model, _kwargs = _get_detector(key)
        onnx_filename = os.path.join(output_dir, re.sub(r'\W+', '-', f'{key}') + '.onnx')
        _model.export_onnx(onnx_filename, **_kwargs)


if __name__ == '__main__':
    cli()
