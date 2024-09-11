import os
import random

from hfutils.operate import get_hf_fs

from benchmark import BaseBenchmark, create_plot_cli
from imgutils.detect import detect_with_booru_yolo

repository = 'deepghs/booru_yolo'
hf_fs = get_hf_fs()
_MODELS = [
    os.path.basename(os.path.dirname(file))
    for file in hf_fs.glob(f'{repository}/*/model.onnx')
]


class BooruYOLODetectBenchmark(BaseBenchmark):
    def __init__(self, model_name: str):
        BaseBenchmark.__init__(self)
        self.model_name = model_name

    def load(self):
        from imgutils.detect.booru_yolo import _open_booru_yolo_model, _get_booru_yolo_labels
        _ = _open_booru_yolo_model(self.model_name)
        _ = _get_booru_yolo_labels(self.model_name)

    def unload(self):
        from imgutils.detect.booru_yolo import _open_booru_yolo_model, _get_booru_yolo_labels
        _open_booru_yolo_model.cache_clear()
        _get_booru_yolo_labels.cache_clear()

    def run(self):
        image_file = random.choice(self.all_images)
        _ = detect_with_booru_yolo(image_file, model_name=self.model_name)


if __name__ == '__main__':
    create_plot_cli(
        [
            (model_name, BooruYOLODetectBenchmark(model_name))
            for model_name in _MODELS
        ],
        title='Benchmark for Anime Booru YOLO Detections',
        run_times=10,
        try_times=20,
    )()
