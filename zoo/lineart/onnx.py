from PIL import Image

from ..utils import get_testfile, onnx_quick_export


class _BaseOnnxMixin:
    def _get_model(self, **kwargs):
        raise NotImplementedError

    def preprocess(self, input_image, **kwargs):
        raise NotImplementedError

    def dynamic_axes(self):
        return {
            "input": {0: "batch", 2: 'height', 3: 'width'},
            "output": {0: "batch", 2: 'height', 3: 'width'},
        }

    def export_onnx(self, onnx_filename, opset_version: int = 14, verbose: bool = True,
                    no_optimize: bool = False, input_image=None, **kwargs):
        input_image = Image.open(input_image or get_testfile('6125785.png'))
        onnx_quick_export(
            self._get_model(**kwargs), self.preprocess(input_image, **kwargs),
            onnx_filename, opset_version, verbose, no_optimize, self.dynamic_axes(), no_gpu=True
        )


class _FixedOnnxMixin(_BaseOnnxMixin):
    def __init__(self, model):
        self._model = model

    def _get_model(self, **kwargs):
        return self._model
