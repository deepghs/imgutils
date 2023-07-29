import random

from benchmark import BaseBenchmark, create_plot_cli
from imgutils.detect import detect_hands


class HandDetectBenchmark(BaseBenchmark):
    def __init__(self, version, level):
        BaseBenchmark.__init__(self)
        self.version = version
        self.level = level

    def load(self):
        from imgutils.detect.hand import _open_hand_detect_model
        _ = _open_hand_detect_model(version=self.version, level=self.level)

    def unload(self):
        from imgutils.detect.hand import _open_hand_detect_model
        _open_hand_detect_model.cache_clear()

    def run(self):
        image_file = random.choice(self.all_images)
        _ = detect_hands(image_file, version=self.version, level=self.level)


if __name__ == '__main__':
    create_plot_cli(
        [
            ('hand v0.8 (yolov8s)', HandDetectBenchmark('v0.8', 's')),
            ('hand v1.0 (yolov8s)', HandDetectBenchmark('v1.0', 's')),
            ('hand v1.0 (yolov8n)', HandDetectBenchmark('v1.0', 'n')),
        ],
        title='Benchmark for Anime Hand Detections',
        run_times=10,
        try_times=20,
    )()
