import random

from PIL import Image

from benchmark import BaseBenchmark, create_plot_cli
from imgutils.ocr.recognize import _text_recognize, _list_rec_models


class OCRRecognizeBenchmark(BaseBenchmark):
    def __init__(self, model):
        BaseBenchmark.__init__(self)
        self.model = model

    def load(self):
        from imgutils.ocr.recognize import _open_ocr_recognition_model
        from imgutils.ocr.recognize import _open_ocr_recognition_dictionary
        _ = _open_ocr_recognition_model(self.model)
        _ = _open_ocr_recognition_dictionary(self.model)

    def unload(self):
        from imgutils.ocr.recognize import _open_ocr_recognition_model
        from imgutils.ocr.recognize import _open_ocr_recognition_dictionary
        _open_ocr_recognition_model.cache_clear()
        _open_ocr_recognition_dictionary.cache_clear()

    def run(self):
        height = 48
        width = int(random.random() * 20 + 10) * 24
        img = Image.new('RGB', (width, height))
        _ = _text_recognize(img, model=self.model)


if __name__ == '__main__':
    create_plot_cli(
        [
            (model, OCRRecognizeBenchmark(model))
            for model in _list_rec_models()
            if 'server' not in model
        ],
        title='Benchmark for OCR Recognitions',
        run_times=10,
        try_times=20,
    )()
