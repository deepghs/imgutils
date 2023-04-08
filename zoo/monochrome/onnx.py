import os
import tempfile

import onnx
import torch
from PIL import Image
from torch import nn

from .dataset import TRANSFORM2_VAL
from .encode import image_encode
from ..utils import get_testfile, onnx_optimize


class ModelWithSoftMax(nn.Module):
    def __init__(self, model):
        nn.Module.__init__(self)
        self.model = model

    def forward(self, x):
        x = self.model(x)
        x = torch.softmax(x, dim=1)
        return x


def export_model_to_onnx(model, onnx_filename, opset_version: int = 14, verbose: bool = True,
                         no_optimize: bool = False, feature_bins: int = 180):
    image = Image.open(get_testfile('6125785.jpg')).convert('RGB')
    if getattr(model, '__dims__', 1) == 1:
        example_input = image_encode(image, bins=feature_bins, normalize=True).float().unsqueeze(0)
    else:
        example_input = TRANSFORM2_VAL(image).float().unsqueeze(0)
    model = ModelWithSoftMax(model).float()

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
            output_names=["output"],

            opset_version=opset_version,
            dynamic_axes={
                "input": {0: "batch"},
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
