import random

from benchmark import BaseBenchmark, create_plot_cli
from imgutils.data import load_image
from imgutils.upscale import real_esrgan_upscape_4x
from imgutils.upscale.real_esrgan import _MODELS


class RealESRGANBenchmark(BaseBenchmark):
    def __init__(self, model: str, size: int):
        BaseBenchmark.__init__(self)
        self.model = model
        self.size = size

    def load(self):
        from imgutils.upscale.real_esrgan import _open_real_esrgan_model
        _ = _open_real_esrgan_model(self.model)

    def unload(self):
        from imgutils.upscale.real_esrgan import _open_real_esrgan_model
        _open_real_esrgan_model.cache_clear()

    def run(self):
        image_file = random.choice(self.all_images)
        image = load_image(image_file, mode='RGB').resize((self.size, self.size))
        real_esrgan_upscape_4x(image, model=self.model)


if __name__ == '__main__':
    create_plot_cli(
        [
            (
                f'{model} ({size}x{size})',
                RealESRGANBenchmark(model, size),
            )
            for size in (224, 384, 512)
            for model in _MODELS
        ],
        title='Benchmark for Real ESRGAN Models',
        run_times=10,
        try_times=20,
    )()
