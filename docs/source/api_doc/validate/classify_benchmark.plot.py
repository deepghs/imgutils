import random

from benchmark import BaseBenchmark, create_plot_cli
from imgutils.validate import anime_classify
from imgutils.validate.classify import _MODEL_NAMES


class AnimeClassifyBenchmark(BaseBenchmark):
    def __init__(self, model):
        BaseBenchmark.__init__(self)
        self.model = model

    def load(self):
        from imgutils.validate.classify import _open_anime_classify_model
        _ = _open_anime_classify_model(self.model)

    def unload(self):
        from imgutils.validate.classify import _open_anime_classify_model
        _open_anime_classify_model.cache_clear()

    def run(self):
        image_file = random.choice(self.all_images)
        _ = anime_classify(image_file, self.model)


if __name__ == '__main__':
    create_plot_cli(
        [
            (name, AnimeClassifyBenchmark(name))
            for name in _MODEL_NAMES
        ],
        title='Benchmark for Anime Classify Models',
        run_times=10,
        try_times=20,
    )()
