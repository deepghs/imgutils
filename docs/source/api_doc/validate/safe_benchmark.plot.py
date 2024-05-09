import os
import random

from huggingface_hub import HfFileSystem
from natsort import natsorted

from benchmark import BaseBenchmark, create_plot_cli
from imgutils.validate import safe_check

hf_fs = HfFileSystem()

REPOSITORY = 'mf666/shit-checker'
MODELS = natsorted([
    os.path.splitext(os.path.relpath(file, REPOSITORY))[0]
    for file in hf_fs.glob(f'{REPOSITORY}/*.onnx')
])


class SafeCheckBenchmark(BaseBenchmark):
    def __init__(self, model):
        BaseBenchmark.__init__(self)
        self.model = model

    def load(self):
        from imgutils.validate.safe import _open_model
        _ = _open_model(self.model)

    def unload(self):
        from imgutils.validate.safe import _open_model
        _open_model.cache_clear()

    def run(self):
        image_file = random.choice(self.all_images)
        _ = safe_check(image_file, self.model)


if __name__ == '__main__':
    create_plot_cli(
        [
            (name, SafeCheckBenchmark(name))
            for name in MODELS
        ],
        title='Benchmark for Safe Check Models',
        run_times=10,
        try_times=20,
    )()
