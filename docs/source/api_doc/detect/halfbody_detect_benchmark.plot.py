import random

from benchmark import BaseBenchmark, create_plot_cli
from imgutils.detect import detect_halfbody


class HalfBodyDetectBenchmark(BaseBenchmark):
    def __init__(self, level, version):
        BaseBenchmark.__init__(self)
        self.level = level
        self.version = version

    def load(self):
        from imgutils.detect.halfbody import _open_halfbody_detect_model
        _ = _open_halfbody_detect_model(level=self.level, version=self.version)

    def unload(self):
        from imgutils.detect.halfbody import _open_halfbody_detect_model
        _open_halfbody_detect_model.cache_clear()

    def run(self):
        image_file = random.choice(self.all_images)
        _ = detect_halfbody(image_file, level=self.level, version=self.version)


if __name__ == '__main__':
    create_plot_cli(
        [
            ('halfbody v1.0 (yolov8s)', HalfBodyDetectBenchmark('s', 'v1.0')),
            ('halfbody v1.0 (yolov8n)', HalfBodyDetectBenchmark('n', 'v1.0')),
            ('halfbody v0.4 (yolov8s)', HalfBodyDetectBenchmark('s', 'v0.4')),
            ('halfbody v0.3 (yolov8s)', HalfBodyDetectBenchmark('s', 'v0.3')),
        ],
        title='Benchmark for Anime Half Body Detections',
        run_times=10,
        try_times=20,
    )()
