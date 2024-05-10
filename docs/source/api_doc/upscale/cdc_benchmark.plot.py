import os.path
import random

from huggingface_hub import HfFileSystem

from benchmark import BaseBenchmark, create_plot_cli
from imgutils.upscale.cdc import upscale_with_cdc

hf_fs = HfFileSystem()
repository = 'deepghs/cdc_anime_onnx'
_CDC_MODELS = [
    os.path.splitext(os.path.relpath(file, repository))[0]
    for file in hf_fs.glob(f'{repository}/*.onnx')
]


class CDCUpscalerBenchmark(BaseBenchmark):
    def __init__(self, model: str):
        BaseBenchmark.__init__(self)
        self.model = model

    def load(self):
        from imgutils.upscale.cdc import _open_cdc_upscaler_model
        _open_cdc_upscaler_model(self.model)

    def unload(self):
        from imgutils.upscale.cdc import _open_cdc_upscaler_model
        _open_cdc_upscaler_model.cache_clear()

    def run(self):
        image_file = random.choice(self.all_images)
        _ = upscale_with_cdc(image_file, model=self.model)


if __name__ == '__main__':
    create_plot_cli(
        [
            (model, CDCUpscalerBenchmark(model))
            for model in _CDC_MODELS
        ],
        title='Benchmark for CDCUpscaler Models',
        run_times=3,
        try_times=3,
    )()
