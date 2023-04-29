import os.path

from hbutils.system import copy
from hbutils.testing import isolated_directory
from ultralytics import YOLO


def export_yolo_to_onnx(yolo: YOLO, onnx_filename, opset_version: int = 14,
                        no_optimize: bool = False):
    _current_path = os.path.abspath(os.curdir)
    with isolated_directory():
        copy(
            yolo.export(format='onnx', dynamic=True, simplify=not no_optimize, opset=opset_version),
            os.path.join(_current_path, onnx_filename)
        )
