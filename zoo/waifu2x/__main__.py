from functools import partial

import click
from ditk import logging

from .sync import sync_to_huggingface
from ..utils import GLOBAL_CONTEXT_SETTINGS
from ..utils import print_version as _origin_print_version

print_version = partial(_origin_print_version, 'zoo.waifu2x')


@click.group(context_settings={**GLOBAL_CONTEXT_SETTINGS})
@click.option('-v', '--version', is_flag=True,
              callback=print_version, expose_value=False, is_eager=True,
              help="Utils with waifu2x models.")
def cli():
    pass  # pragma: no cover


@cli.command('sync', help='Export feature extract model as onnx.',
             context_settings={**GLOBAL_CONTEXT_SETTINGS})
@click.option('--repository', '-r', 'repository', type=str, default='deepghs/waifu2x_onnx',
              help='Repository to sync.', show_default=True)
def sync(repository: str):
    logging.try_init_root(logging.INFO)
    sync_to_huggingface(repository)


if __name__ == '__main__':
    cli()
