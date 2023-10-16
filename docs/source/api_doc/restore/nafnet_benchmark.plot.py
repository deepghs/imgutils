import random

from benchmark import BaseBenchmark, create_plot_cli
from imgutils.restore.nafnet import NafNetModelTyping, restore_with_nafnet


class NafNetBenchmark(BaseBenchmark):
    def __init__(self, model: NafNetModelTyping):
        BaseBenchmark.__init__(self)
        self.model = model

    def load(self):
        from imgutils.restore.nafnet import _open_nafnet_model
        _open_nafnet_model(self.model)

    def unload(self):
        from imgutils.restore.nafnet import _open_nafnet_model
        _open_nafnet_model.cache_clear()

    def run(self):
        image_file = random.choice(self.all_images)
        _ = restore_with_nafnet(image_file, model=self.model)


if __name__ == '__main__':
    create_plot_cli(
        [
            ('NafNet-REDS', NafNetBenchmark('REDS')),
            ('NafNet-GoPro', NafNetBenchmark('GoPro')),
            ('NafNet-SIDD', NafNetBenchmark('SIDD')),
        ],
        title='Benchmark for NafNet Models',
        run_times=5,
        try_times=10,
    )()
