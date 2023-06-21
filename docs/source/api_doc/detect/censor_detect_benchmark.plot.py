import random

from benchmark import BaseBenchmark, create_plot_cli
from imgutils.detect import detect_censors


class CensorDetectBenchmark(BaseBenchmark):
    def __init__(self, level, version):
        BaseBenchmark.__init__(self)
        self.level = level
        self.version = version

    def load(self):
        from imgutils.detect.censor import _open_censor_detect_model
        _ = _open_censor_detect_model(level=self.level, version=self.version)

    def unload(self):
        from imgutils.detect.censor import _open_censor_detect_model
        _open_censor_detect_model.cache_clear()

    def run(self):
        image_file = random.choice(self.all_images)
        _ = detect_censors(image_file, level=self.level, version=self.version)


if __name__ == '__main__':
    create_plot_cli(
        [
            ('censor v1 (yolov8s)', CensorDetectBenchmark('s', 'v1')),
            ('censor v1 (yolov8n)', CensorDetectBenchmark('n', 'v1')),
            ('censor v0.10 (yolov8s)', CensorDetectBenchmark('s', 'v0.10')),
            ('censor v0.9 (yolov8s)', CensorDetectBenchmark('s', 'v0.9')),
            # ('censor v0.8 (yolov8s)', CensorDetectBenchmark('s', 'v0.8')),
            # ('censor v0.7 (yolov8s)', CensorDetectBenchmark('s', 'v0.7')),
        ],
        title='Benchmark for Anime Censor Detections',
        run_times=10,
        try_times=20,
    )()
