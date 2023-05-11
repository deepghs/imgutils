import random

from benchmark import BaseBenchmark, create_plot_cli
from imgutils.detect import detect_person


class PersonDetectBenchmark(BaseBenchmark):
    def __init__(self, level):
        BaseBenchmark.__init__(self)
        self.level = level

    def load(self):
        from imgutils.detect.person import _open_person_detect_model
        _ = _open_person_detect_model(level=self.level)

    def unload(self):
        from imgutils.detect.person import _open_person_detect_model
        _open_person_detect_model.cache_clear()

    def run(self):
        image_file = random.choice(self.all_images)
        _ = detect_person(image_file, level=self.level)


if __name__ == '__main__':
    create_plot_cli(
        [
            ('person (yolov8s)', PersonDetectBenchmark('s')),
            ('person (yolov8m)', PersonDetectBenchmark('m')),
            ('person (yolov8x)', PersonDetectBenchmark('x')),
        ],
        title='Benchmark for Anime Person Detections',
        run_times=10,
        try_times=20
        figsize=(1080, 600)
    )()
