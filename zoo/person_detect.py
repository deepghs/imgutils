import os.path
from functools import partial
from typing import List

import click
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm
from ultralytics import YOLO

from .detection.onnx import export_yolo_to_onnx
from .utils import GLOBAL_CONTEXT_SETTINGS
from .utils import print_version as _origin_print_version

print_version = partial(_origin_print_version, 'zoo.person_detect')


@click.group(context_settings={**GLOBAL_CONTEXT_SETTINGS})
@click.option('-v', '--version', is_flag=True,
              callback=print_version, expose_value=False, is_eager=True,
              help="Show version information.")
def cli():
    pass  # pragma: no cover


_KNOWN_CKPTS: List[str] = [
    # 'person_detect_best_s.pt',
    # 'person_detect_best_m.pt',
    # 'person_detect_best_x.pt',
    # 'person_detect_plus_best_m.pt',
    # 'person_detect_plus_v1.1_best_m.pt',
    'person_detect_plus_v1.1_best_s.pt',
    'person_detect_plus_v1.1_best_n.pt',
]


@cli.command('export', help='Export all models as onnx.',
             context_settings={**GLOBAL_CONTEXT_SETTINGS})
@click.option('--output_dir', '-O', 'output_dir', type=click.Path(file_okay=False), required=True,
              help='Output directory of all models.', show_default=True)
def export(output_dir: str):
    for ckpt in tqdm(_KNOWN_CKPTS):
        yolo = YOLO(hf_hub_download('deepghs/imgutils-models', f'person_detect/{ckpt}'))
        filebody, _ = os.path.splitext(ckpt)
        output_file = os.path.join(output_dir, f'{filebody}.onnx')
        export_yolo_to_onnx(yolo, output_file)


if __name__ == '__main__':
    cli()
