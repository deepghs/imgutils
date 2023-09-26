import random

from benchmark import BaseBenchmark, create_plot_cli
from imgutils.detect import detect_eyes


class EyeDetectBenchmark(BaseBenchmark):
    def __init__(self, level, version):
        BaseBenchmark.__init__(self)
        self.level = level
        self.version = version

    def load(self):
        from imgutils.detect.eye import _open_eye_detect_model
        _ = _open_eye_detect_model(level=self.level, version=self.version)

    def unload(self):
        from imgutils.detect.eye import _open_eye_detect_model
        _open_eye_detect_model.cache_clear()

    def run(self):
        image_file = random.choice(self.all_images)
        _ = detect_eyes(image_file, level=self.level, version=self.version)


if __name__ == '__main__':
    create_plot_cli(
        [
            ('eye v1.0 (yolov8s)', EyeDetectBenchmark('s', 'v1.0')),
            ('eye v1.0 (yolov8n)', EyeDetectBenchmark('n', 'v1.0')),
            ('eye v0.8 (yolov8s)', EyeDetectBenchmark('s', 'v0.8')),
            ('eye v0.7 (yolov8s)', EyeDetectBenchmark('s', 'v0.7')),
        ],
        title='Benchmark for Anime Eyes Detections',
        run_times=10,
        try_times=20,
    )()
