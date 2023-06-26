import random

from benchmark import BaseBenchmark, create_plot_cli
from imgutils.validate import nsfw_pred_score
from imgutils.validate.nsfw import _MODEL_NAMES


class AnimeNSFWBenchmark(BaseBenchmark):
    def __init__(self, model):
        BaseBenchmark.__init__(self)
        self.model = model

    def load(self):
        from imgutils.validate.nsfw import _open_nsfw_model
        _ = _open_nsfw_model(self.model)

    def unload(self):
        from imgutils.validate.nsfw import _open_nsfw_model
        _open_nsfw_model.cache_clear()

    def run(self):
        image_file = random.choice(self.all_images)
        _ = nsfw_pred_score(image_file, self.model)


if __name__ == '__main__':
    create_plot_cli(
        [
            (name, AnimeNSFWBenchmark(name))
            for name in _MODEL_NAMES
        ],
        title='Benchmark for NSFW Models',
        run_times=10,
        try_times=20,
    )()
