import random

from benchmark import BaseBenchmark, create_plot_cli
from imgutils.generic.classify import _open_models_for_repo_id
from imgutils.validate import anime_dbrating
from imgutils.validate.dbrating import _REPO_ID

_MODEL_NAMES = _open_models_for_repo_id(_REPO_ID).model_names


class AnimeDBRatingCharacterBenchmark(BaseBenchmark):
    def __init__(self, model):
        BaseBenchmark.__init__(self)
        self.model = model

    def load(self):
        _open_models_for_repo_id(_REPO_ID)._open_model(self.model)

    def unload(self):
        _open_models_for_repo_id(_REPO_ID).clear()

    def run(self):
        image_file = random.choice(self.all_images)
        _ = anime_dbrating(image_file, self.model)


if __name__ == '__main__':
    create_plot_cli(
        [
            (name, AnimeDBRatingCharacterBenchmark(name))
            for name in _MODEL_NAMES
        ],
        title='Benchmark for Danbooru Rating Models',
        run_times=10,
        try_times=20,
    )()
