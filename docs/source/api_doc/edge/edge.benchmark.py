import random

from benchmark import BaseBenchmark, create_plot_cli
from imgutils.edge import get_edge_by_canny, get_edge_by_lineart_anime, get_edge_by_lineart


class CannyBenchmark(BaseBenchmark):
    def load(self):
        pass

    def unload(self):
        pass

    def run(self):
        image_file = random.choice(self.all_images)
        _ = get_edge_by_canny(image_file)


class LineartAnimeBenchmark(BaseBenchmark):
    def load(self):
        from imgutils.edge.lineart_anime import _open_la_anime_model
        _ = _open_la_anime_model()

    def unload(self):
        from imgutils.edge.lineart_anime import _open_la_anime_model
        _open_la_anime_model.cache_clear()

    def run(self):
        image_file = random.choice(self.all_images)
        _ = get_edge_by_lineart_anime(image_file)


class LineartBenchmark(BaseBenchmark):
    def __init__(self, coarse: bool):
        BaseBenchmark.__init__(self)
        self.coarse = coarse

    def load(self):
        from imgutils.edge.lineart import _open_la_model
        _ = _open_la_model(self.coarse)

    def unload(self):
        from imgutils.edge.lineart import _open_la_model
        _open_la_model.cache_clear()

    def run(self):
        image_file = random.choice(self.all_images)
        _ = get_edge_by_lineart(image_file, coarse=self.coarse)


if __name__ == '__main__':
    create_plot_cli(
        [
            ('canny', CannyBenchmark()),
            ('lineart', LineartBenchmark(coarse=False)),
            ('lineart (coarse)', LineartBenchmark(coarse=True)),
            ('lineart-anime', LineartAnimeBenchmark()),
        ],
        title='Benchmark for Edge Models',
        run_times=10,
        try_times=20,
    )()
