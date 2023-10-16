import random

from benchmark import BaseBenchmark, create_plot_cli
from imgutils.ocr.detect import _detect_text, _list_det_models


class OCRDetectBenchmark(BaseBenchmark):
    def __init__(self, model):
        BaseBenchmark.__init__(self)
        self.model = model

    def load(self):
        from imgutils.ocr.detect import _open_ocr_detection_model
        _ = _open_ocr_detection_model(self.model)

    def unload(self):
        from imgutils.ocr.detect import _open_ocr_detection_model
        _open_ocr_detection_model.cache_clear()

    def run(self):
        image_file = random.choice(self.all_images)
        _ = _detect_text(image_file, model=self.model)


if __name__ == '__main__':
    create_plot_cli(
        [
            (model, OCRDetectBenchmark(model))
            for model in _list_det_models()
        ],
        title='Benchmark for OCR Detections',
        run_times=10,
        try_times=20,
    )()
