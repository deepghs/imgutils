import random

from benchmark import BaseBenchmark, create_plot_cli
from imgutils.pose import dwpose_estimate


class DWPoseBenchmark(BaseBenchmark):
    def __init__(self, auto_detect: bool = False):
        BaseBenchmark.__init__(self)
        self.auto_detect = auto_detect

    def load(self):
        from imgutils.pose.dwpose import _open_dwpose_model
        _ = _open_dwpose_model()

    def unload(self):
        from imgutils.pose.dwpose import _open_dwpose_model
        _open_dwpose_model.cache_clear()

    def run(self):
        image_file = random.choice(self.all_images)
        dwpose_estimate(image_file, auto_detect=self.auto_detect)


if __name__ == '__main__':
    create_plot_cli(
        [
            ('DWPose', DWPoseBenchmark(auto_detect=True)),
            ('DWPose (Single)', DWPoseBenchmark(auto_detect=True)),
        ],
        title='Benchmark for DWPose',
        run_times=10,
        try_times=20,
    )()
