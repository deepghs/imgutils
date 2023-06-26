import random

from benchmark import BaseBenchmark, create_plot_cli
from imgutils.validate import anime_rating
from imgutils.validate.rating import _MODEL_NAMES


class AnimeRatingBenchmark(BaseBenchmark):
    def __init__(self, model):
        BaseBenchmark.__init__(self)
        self.model = model

    def load(self):
        from imgutils.validate.rating import _open_anime_rating_model
        _ = _open_anime_rating_model(self.model)

    def unload(self):
        from imgutils.validate.rating import _open_anime_rating_model
        _open_anime_rating_model.cache_clear()

    def run(self):
        image_file = random.choice(self.all_images)
        _ = anime_rating(image_file, self.model)


if __name__ == '__main__':
    create_plot_cli(
        [
            (name, AnimeRatingBenchmark(name))
            for name in _MODEL_NAMES
        ],
        title='Benchmark for Anime Rating Models',
        run_times=10,
        try_times=20,
    )()
