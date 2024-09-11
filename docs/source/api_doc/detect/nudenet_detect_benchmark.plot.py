import random

from benchmark import BaseBenchmark, create_plot_cli
from imgutils.detect import detect_with_nudenet


class NudenetDetectBenchmark(BaseBenchmark):
    def __init__(self):
        BaseBenchmark.__init__(self)

    def load(self):
        from imgutils.detect.nudenet import _open_nudenet_yolo, _open_nudenet_nms
        _ = _open_nudenet_yolo()
        _ = _open_nudenet_nms()

    def unload(self):
        from imgutils.detect.nudenet import _open_nudenet_yolo, _open_nudenet_nms
        _open_nudenet_yolo.cache_clear()
        _open_nudenet_nms.cache_clear()

    def run(self):
        image_file = random.choice(self.all_images)
        _ = detect_with_nudenet(image_file)


if __name__ == '__main__':
    create_plot_cli(
        [
            ('Nudenet', NudenetDetectBenchmark()),
        ],
        title='Benchmark for Anime NudeNet Detections',
        run_times=10,
        try_times=20,
    )()
