import random

from benchmark import BaseBenchmark, create_plot_cli
from imgutils.detect import detect_person


class PersonDetectBenchmark(BaseBenchmark):
    def __init__(self, level, version):
        BaseBenchmark.__init__(self)
        self.level = level
        self.version = version

    def load(self):
        from imgutils.detect.person import _open_person_detect_model
        _ = _open_person_detect_model(level=self.level, version=self.version)

    def unload(self):
        from imgutils.detect.person import _open_person_detect_model
        _open_person_detect_model.cache_clear()

    def run(self):
        image_file = random.choice(self.all_images)
        _ = detect_person(image_file, level=self.level, version=self.version)


if __name__ == '__main__':
    create_plot_cli(
        [
            ('person v1.1 (yolov8m)', PersonDetectBenchmark('m', 'v1.1')),
            ('person v1 (yolov8m)', PersonDetectBenchmark('m', 'v1')),
            ('person v0 (yolov8s)', PersonDetectBenchmark('s', 'v0')),
            ('person v0 (yolov8m)', PersonDetectBenchmark('m', 'v0')),
            ('person v0 (yolov8x)', PersonDetectBenchmark('x', 'v0')),
        ],
        title='Benchmark for Anime Person Detections',
        run_times=10,
        try_times=20,
    )()
