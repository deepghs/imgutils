import random

from benchmark import BaseBenchmark, create_plot_cli
from imgutils.validate import get_monochrome_score


class MonochromeBenchmark(BaseBenchmark):
    def __init__(self, model, safe):
        BaseBenchmark.__init__(self)
        self.model = model
        self.safe = safe

    def load(self):
        from imgutils.validate.monochrome import _monochrome_validate_model
        _ = _monochrome_validate_model(self.model, self.safe)

    def unload(self):
        from imgutils.validate.monochrome import _monochrome_validate_model
        _monochrome_validate_model.cache_clear()

    def run(self):
        image_file = random.choice(self.all_images)
        _ = get_monochrome_score(image_file, model=self.model, safe=self.safe)


if __name__ == '__main__':
    create_plot_cli(
        [
            ('caformer_s36 (unsafe)', MonochromeBenchmark('caformer_s36', False)),
            ('caformer_s36 (safe)', MonochromeBenchmark('caformer_s36', True)),
            ('mobilenetv3 (unsafe)', MonochromeBenchmark('mobilenetv3', False)),
            ('mobilenetv3 (safe)', MonochromeBenchmark('mobilenetv3', True)),
            ('mobilenetv3_dist (unsafe)', MonochromeBenchmark('mobilenetv3_dist', False)),
            ('mobilenetv3_dist (safe)', MonochromeBenchmark('mobilenetv3_dist', True)),
        ],
        title='Benchmark for Monochrome Check Models',
        run_times=10,
        try_times=20,
    )()
