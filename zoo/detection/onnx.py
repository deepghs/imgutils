import os.path
from shutil import SameFileError

from hbutils.system import copy
from ultralytics import YOLO


def export_yolo_to_onnx(yolo: YOLO, onnx_filename, opset_version: int = 14,
                        no_optimize: bool = False):
    if os.path.dirname(onnx_filename):
        os.makedirs(os.path.dirname(onnx_filename), exist_ok=True)

    _retval = yolo.export(format='onnx', dynamic=True, simplify=not no_optimize, opset=opset_version)
    _exported_onnx_file = _retval or (os.path.splitext(yolo.ckpt_path)[0] + '.onnx')
    try:
        copy(_exported_onnx_file, onnx_filename)
    except SameFileError:
        pass
