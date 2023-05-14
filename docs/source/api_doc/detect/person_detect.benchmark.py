import random

from benchmark import BaseBenchmark, create_plot_cli
from imgutils.detect import detect_person


class PersonDetectBenchmark(BaseBenchmark):
    def __init__(self, level, plus):
        BaseBenchmark.__init__(self)
        self.level = level
        self.plus = plus

    def load(self):
        from imgutils.detect.person import _open_person_detect_model
        _ = _open_person_detect_model(level=self.level, plus=self.plus)

    def unload(self):
        from imgutils.detect.person import _open_person_detect_model
        _open_person_detect_model.cache_clear()

    def run(self):
        image_file = random.choice(self.all_images)
        _ = detect_person(image_file, level=self.level, plus=self.plus)


if __name__ == '__main__':
    create_plot_cli(
        [
            ('person plus (yolov8m)', PersonDetectBenchmark('m', True)),
            ('person (yolov8s)', PersonDetectBenchmark('s', False)),
            ('person (yolov8m)', PersonDetectBenchmark('m', False)),
            ('person (yolov8x)', PersonDetectBenchmark('x', False)),
        ],
        title='Benchmark for Anime Person Detections',
        run_times=10,
        try_times=20,
    )()
