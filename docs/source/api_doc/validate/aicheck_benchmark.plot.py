import random

from benchmark import BaseBenchmark, create_plot_cli
from imgutils.validate import get_ai_created_score
from imgutils.validate.aicheck import _MODEL_NAMES


class AnimeAICheckBenchmark(BaseBenchmark):
    def __init__(self, model):
        BaseBenchmark.__init__(self)
        self.model = model

    def load(self):
        from imgutils.validate.aicheck import _open_anime_aicheck_model
        _ = _open_anime_aicheck_model(self.model)

    def unload(self):
        from imgutils.validate.aicheck import _open_anime_aicheck_model
        _open_anime_aicheck_model.clear()

    def run(self):
        image_file = random.choice(self.all_images)
        _ = get_ai_created_score(image_file, self.model)


if __name__ == '__main__':
    create_plot_cli(
        [
            (name, AnimeAICheckBenchmark(name))
            for name in _MODEL_NAMES
        ],
        title='Benchmark for Anime AI-Check Models',
        run_times=10,
        try_times=20,
    )()
