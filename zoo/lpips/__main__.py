import os.path
from functools import partial

import click
import lpips

from .models import LPIPSFeature, LPIPSDiff
from .onnx import export_feat_model_to_onnx, export_diff_model_to_onnx
from ..utils import GLOBAL_CONTEXT_SETTINGS
from ..utils import print_version as _origin_print_version

print_version = partial(_origin_print_version, 'zoo.lpips')


@click.group(context_settings={**GLOBAL_CONTEXT_SETTINGS})
@click.option('-v', '--version', is_flag=True,
              callback=print_version, expose_value=False, is_eager=True,
              help="Utils with pixiv resources.")
def cli():
    pass  # pragma: no cover


@cli.command('export_feature', help='Export feature extract model as onnx.',
             context_settings={**GLOBAL_CONTEXT_SETTINGS})
@click.option('--output', '-o', 'output', type=click.Path(dir_okay=False), required=True,
              help='Output path of feature model.', show_default=True)
def export_feature(output: str):
    model = lpips.LPIPS(net='alex', spatial=False)
    feature_model = LPIPSFeature(model)
    export_feat_model_to_onnx(feature_model, output)


@cli.command('export_diff', help='Export difference model as onnx.',
             context_settings={**GLOBAL_CONTEXT_SETTINGS})
@click.option('--output', '-o', 'output', type=click.Path(dir_okay=False), required=True,
              help='Output path of diff model.', show_default=True)
def export_diff(output: str):
    model = lpips.LPIPS(net='alex', spatial=False)
    diff_model = LPIPSDiff(model)
    export_diff_model_to_onnx(diff_model, output)


@cli.command('export', help='Export all lpips models as onnx.',
             context_settings={**GLOBAL_CONTEXT_SETTINGS})
@click.option('--output_dir', '-O', 'output_dir', type=click.Path(file_okay=False), required=True,
              help='Output directory of all models.', show_default=True)
def export(output_dir: str):
    model = lpips.LPIPS(net='alex', spatial=False)
    feature_model = LPIPSFeature(model)
    diff_model = LPIPSDiff(model)

    export_feat_model_to_onnx(feature_model, os.path.join(output_dir, 'lpips_feature.onnx'))
    export_diff_model_to_onnx(diff_model, os.path.join(output_dir, 'lpips_diff.onnx'))


if __name__ == '__main__':
    cli()
