import os.path

from hbutils.system import copy
from ultralytics import YOLO


def export_yolo_to_onnx(yolo: YOLO, onnx_filename, opset_version: int = 14,
                        no_optimize: bool = False):
    if os.path.dirname(onnx_filename):
        os.makedirs(os.path.dirname(onnx_filename), exist_ok=True)
    copy(
        yolo.export(format='onnx', dynamic=True, simplify=not no_optimize, opset=opset_version),
        onnx_filename
    )
