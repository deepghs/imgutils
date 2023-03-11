import os
import tempfile

import lpips
import onnx
import torch
from PIL import Image

from .dispatch import _TRANSFORM
from .models import LPIPSFeature
from ..utils import get_testfile, onnx_optimize


def export_feat_model_to_onnx(model, onnx_filename, opset_version: int = 14, verbose: bool = True,
                              no_optimize: bool = False):
    image = Image.open(get_testfile('6125785.jpg')).convert('RGB')
    example_input = _TRANSFORM(image).unsqueeze(0)

    if torch.cuda.is_available():
        example_input = example_input.cuda()
        model = model.cuda()

    with torch.no_grad(), tempfile.TemporaryDirectory() as td:
        onnx_model_file = os.path.join(td, 'model.onnx')
        torch.onnx.export(
            model,
            example_input,
            onnx_model_file,
            verbose=verbose,
            input_names=["input"],
            output_names=["feat_0", "feat_1", "feat_2", "feat_3", "feat_4"],

            opset_version=opset_version,
            dynamic_axes={
                "input": {0: "batch"},
                "feat_0": {0: "batch"},
                "feat_1": {0: "batch"},
                "feat_2": {0: "batch"},
                "feat_3": {0: "batch"},
                "feat_4": {0: "batch"},
            }
        )

        model = onnx.load(onnx_model_file)
        if not no_optimize:
            model = onnx_optimize(model)

        output_model_dir, _ = os.path.split(onnx_filename)
        if output_model_dir:
            os.makedirs(output_model_dir, exist_ok=True)
        onnx.save(model, onnx_filename)


def export_diff_model_to_onnx(model, onnx_filename, opset_version: int = 14, verbose: bool = True,
                              no_optimize: bool = False):
    image = Image.open(get_testfile('6125785.jpg')).convert('RGB')
    example_input = _TRANSFORM(image).unsqueeze(0)
    with torch.no_grad():
        _lpips_model = lpips.LPIPS(net='alex', spatial=False)
        feature_model = LPIPSFeature(_lpips_model)
        feats = feature_model(example_input)

    all_feats = tuple([*feats, *feats])

    if torch.cuda.is_available():
        all_feats = [item.cuda() for item in all_feats]
        model = model.cuda()

    with torch.no_grad(), tempfile.TemporaryDirectory() as td:
        onnx_model_file = os.path.join(td, 'model.onnx')
        torch.onnx.export(
            model,
            tuple(all_feats),
            onnx_model_file,
            verbose=verbose,
            input_names=[
                "feat_x_0", "feat_x_1", "feat_x_2", "feat_x_3", "feat_x_4",
                "feat_y_0", "feat_y_1", "feat_y_2", "feat_y_3", "feat_y_4",
            ],
            output_names=["output"],

            opset_version=opset_version,
            dynamic_axes={
                "feat_x_0": {0: "batch"},
                "feat_x_1": {0: "batch"},
                "feat_x_2": {0: "batch"},
                "feat_x_3": {0: "batch"},
                "feat_x_4": {0: "batch"},
                "feat_y_0": {0: "batch"},
                "feat_y_1": {0: "batch"},
                "feat_y_2": {0: "batch"},
                "feat_y_3": {0: "batch"},
                "feat_y_4": {0: "batch"},
                "output": {0: "batch"},
            }
        )

        model = onnx.load(onnx_model_file)
        if not no_optimize:
            model = onnx_optimize(model)

        output_model_dir, _ = os.path.split(onnx_filename)
        if output_model_dir:
            os.makedirs(output_model_dir, exist_ok=True)
        onnx.save(model, onnx_filename)
