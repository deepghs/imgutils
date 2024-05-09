import random

from benchmark import BaseBenchmark, create_plot_cli
from imgutils.metrics import anime_dbaesthetic
from imgutils.metrics.dbaesthetic import _MODEL

_MODEL_NAMES = _MODEL.classifier.model_names


class DBAestheticBenchmark(BaseBenchmark):
    def __init__(self, model_name: str):
        BaseBenchmark.__init__(self)
        self.model_name = model_name

    def load(self):
        from imgutils.metrics.dbaesthetic import _MODEL
        _MODEL.classifier._open_model(self.model_name)
        _MODEL._get_xy_samples(self.model_name)

    def unload(self):
        from imgutils.metrics.dbaesthetic import _MODEL
        _MODEL.clear()

    def run(self):
        image_file = random.choice(self.all_images)
        _ = anime_dbaesthetic(image_file)


if __name__ == '__main__':
    create_plot_cli(
        [
            (model_name, DBAestheticBenchmark(model_name))
            for model_name in _MODEL_NAMES
        ],
        title='Benchmark for Danbooru-Based Aesthetic Models',
        run_times=10,
        try_times=20,
    )()
