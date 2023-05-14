import random

from benchmark import BaseBenchmark, create_plot_cli
from imgutils.metrics import get_aesthetic_score


class AestheticBenchmark(BaseBenchmark):
    def load(self):
        from imgutils.metrics.aesthetic import _open_aesthetic_model
        _ = _open_aesthetic_model()

    def unload(self):
        from imgutils.metrics.aesthetic import _open_aesthetic_model
        _open_aesthetic_model.cache_clear()

    def run(self):
        image_file = random.choice(self.all_images)
        _ = get_aesthetic_score(image_file)


if __name__ == '__main__':
    create_plot_cli(
        [
            ('aesthetic', AestheticBenchmark()),
        ],
        title='Benchmark for Aesthetic Models',
        run_times=10,
        try_times=20,
    )()
