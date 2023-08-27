import random

from benchmark import BaseBenchmark, create_plot_cli
from imgutils.validate import anime_teen
from imgutils.validate.teen import _MODEL_NAMES


class AnimeTeenBenchmark(BaseBenchmark):
    def __init__(self, model):
        BaseBenchmark.__init__(self)
        self.model = model

    def load(self):
        from imgutils.validate.teen import _open_anime_teen_model
        _ = _open_anime_teen_model(self.model)

    def unload(self):
        from imgutils.validate.teen import _open_anime_teen_model
        _open_anime_teen_model.cache_clear()

    def run(self):
        image_file = random.choice(self.all_images)
        _ = anime_teen(image_file, self.model)


if __name__ == '__main__':
    create_plot_cli(
        [
            (name, AnimeTeenBenchmark(name))
            for name in _MODEL_NAMES
        ],
        title='Benchmark for Anime teen Models',
        run_times=10,
        try_times=20,
    )()
