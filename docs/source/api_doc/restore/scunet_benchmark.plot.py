import random

from benchmark import BaseBenchmark, create_plot_cli
from imgutils.restore.scunet import SCUNetModelTyping, restore_with_scunet


class SCUNetBenchmark(BaseBenchmark):
    def __init__(self, model: SCUNetModelTyping):
        BaseBenchmark.__init__(self)
        self.model = model

    def load(self):
        from imgutils.restore.scunet import _open_scunet_model
        _open_scunet_model(self.model)

    def unload(self):
        from imgutils.restore.scunet import _open_scunet_model
        _open_scunet_model.cache_clear()

    def run(self):
        image_file = random.choice(self.all_images)
        _ = restore_with_scunet(image_file, model=self.model)


if __name__ == '__main__':
    create_plot_cli(
        [
            ('SCUNet-GAN', SCUNetBenchmark('GAN')),
            ('SCUNet-PSNR', SCUNetBenchmark('PSNR')),
        ],
        title='Benchmark for SCUNet Models',
        run_times=5,
        try_times=10,
    )()
