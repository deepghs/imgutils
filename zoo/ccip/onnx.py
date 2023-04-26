import os
from tempfile import TemporaryDirectory

import onnx
import torch
from PIL import Image
from torch import nn

from .model import CCIP
from ..utils import get_testfile, onnx_optimize


class ModelWithSoftMax(nn.Module):
    def __init__(self, model):
        nn.Module.__init__(self)
        self.model = model

    def forward(self, x):
        x = self.model(x)
        x = torch.softmax(x, dim=-1)
        return x


def get_batch_images(preprocess) -> torch.Tensor:
    image_files = [
        get_testfile('6124220.jpg'),
        get_testfile('6125785.jpg'),
        get_testfile('6125901.jpg'),
    ]

    return torch.stack([
        preprocess(Image.open(img))
        for img in image_files
    ])


def _onnx_export(model, example_input, onnx_filename, opset_version: int = 14, verbose: bool = True,
                 no_optimize: bool = False, dynamic_axes=None):
    model = model.float()
    if torch.cuda.is_available():
        example_input = example_input.cuda()
        model = model.cuda()
    else:
        example_input = example_input.cpu()
        model = model.cpu()

    with torch.no_grad(), TemporaryDirectory() as td:
        onnx_model_file = os.path.join(td, 'model.onnx')
        torch.onnx.export(
            model,
            example_input,
            onnx_model_file,
            verbose=verbose,
            input_names=["input"],
            output_names=["output"],

            opset_version=opset_version,
            dynamic_axes=dynamic_axes or {},
        )

        model = onnx.load(onnx_model_file)
        if not no_optimize:
            model = onnx_optimize(model)

        output_model_dir, _ = os.path.split(onnx_filename)
        if output_model_dir:
            os.makedirs(output_model_dir, exist_ok=True)
        onnx.save(model, onnx_filename)


def export_full_model_to_onnx(model: CCIP, onnx_filename, opset_version: int = 14, verbose: bool = True,
                              no_optimize: bool = False):
    example_input = get_batch_images(model.preprocess)
    return _onnx_export(
        ModelWithSoftMax(model), example_input,
        onnx_filename, opset_version, verbose, no_optimize,
        dynamic_axes={
            "input": {0: "batch"},
            "output": {0: "batch", 1: "batch"},
        }
    )


def export_feat_model_to_onnx(model: CCIP, onnx_filename, opset_version: int = 14, verbose: bool = True,
                              no_optimize: bool = False):
    example_input = get_batch_images(model.preprocess)
    return _onnx_export(
        model.feature, example_input,
        onnx_filename, opset_version, verbose, no_optimize,
        dynamic_axes={
            "input": {0: "batch"},
            "output": {0: "batch"},
        }
    )


def export_metrics_model_to_onnx(model: CCIP, onnx_filename, opset_version: int = 14, verbose: bool = True,
                                 no_optimize: bool = False):
    origin = get_batch_images(model.preprocess)
    with torch.no_grad():
        example_input = model.feature(origin)

    return _onnx_export(
        ModelWithSoftMax(model.metrics), example_input,
        onnx_filename, opset_version, verbose, no_optimize,
        dynamic_axes={
            "input": {0: "batch"},
            "output": {0: "batch", 1: "batch"},
        }
    )
