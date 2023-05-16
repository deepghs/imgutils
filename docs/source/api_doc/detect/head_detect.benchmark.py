import random

from benchmark import BaseBenchmark, create_plot_cli
from imgutils.detect import detect_heads


class HeadDetectBenchmark(BaseBenchmark):
    def __init__(self, level):
        BaseBenchmark.__init__(self)
        self.level = level

    def load(self):
        from imgutils.detect.head import _open_head_detect_model
        _ = _open_head_detect_model(level=self.level)

    def unload(self):
        from imgutils.detect.head import _open_head_detect_model
        _open_head_detect_model.cache_clear()

    def run(self):
        image_file = random.choice(self.all_images)
        _ = detect_heads(image_file, level=self.level)


if __name__ == '__main__':
    create_plot_cli(
        [
            ('head (yolov8s)', HeadDetectBenchmark('s')),
            ('head (yolov8n)', HeadDetectBenchmark('n')),
        ],
        title='Benchmark for Anime Head Detections',
        run_times=10,
        try_times=5,
    )()
