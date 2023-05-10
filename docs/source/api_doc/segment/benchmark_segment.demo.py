import random

from benchmark import BaseBenchmark, create_plot
from imgutils.segment import get_isnetis_mask


class IsnetisBenchmark(BaseBenchmark):
    def load(self):
        from imgutils.segment.isnetis import _get_model
        _ = _get_model()

    def unload(self):
        from imgutils.segment.isnetis import _get_model
        _get_model.cache_clear()

    def run(self):
        image_file = random.choice(self.all_images)
        _ = get_isnetis_mask(image_file)


if __name__ == '__main__':
    create_plot(
        [
            ('isnetis', IsnetisBenchmark()),
        ],
        save_as='benchmark_segment.dat.svg',
        title='Benchmark for Segment Models',
        run_times=10,
        try_times=5,
        figsize=(1080, 600)
    )
