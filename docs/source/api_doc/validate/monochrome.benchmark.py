import random

from benchmark import BaseBenchmark, create_plot_cli
from imgutils.validate import get_monochrome_score
from imgutils.validate.monochrome import _MODELS


class MonochromeBenchmark(BaseBenchmark):
    def __init__(self, safe=0):
        BaseBenchmark.__init__(self)
        self.safe = safe

    def load(self):
        from imgutils.validate.monochrome import _monochrome_validate_model
        _ = _monochrome_validate_model(_MODELS[self.safe])

    def unload(self):
        from imgutils.validate.monochrome import _monochrome_validate_model
        _monochrome_validate_model.cache_clear()

    def run(self):
        image_file = random.choice(self.all_images)
        _ = get_monochrome_score(image_file, safe=self.safe)


if __name__ == '__main__':
    create_plot_cli(
        [
            ('monochrome', MonochromeBenchmark()),
            ('monochrome (safe 2)', MonochromeBenchmark(2)),
            ('monochrome (safe 4)', MonochromeBenchmark(4)),
        ],
        title='Benchmark for Monochrome Check Models',
        run_times=10,
        try_times=20,
    )()
