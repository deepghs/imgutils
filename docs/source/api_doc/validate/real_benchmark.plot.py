import os
import random

from huggingface_hub import HfFileSystem

from benchmark import BaseBenchmark, create_plot_cli
from imgutils.validate import anime_real

hf_fs = HfFileSystem()

_REPOSITORY = 'deepghs/anime_real_cls'
_MODEL_NAMES = [
    os.path.relpath(file, _REPOSITORY).split('/')[0] for file in
    hf_fs.glob(f'{_REPOSITORY}/*/model.onnx')
]


class AnimeRealBenchmark(BaseBenchmark):
    def __init__(self, model):
        BaseBenchmark.__init__(self)
        self.model = model

    def load(self):
        from imgutils.validate.real import _open_anime_real_model
        _ = _open_anime_real_model(self.model)

    def unload(self):
        from imgutils.validate.real import _open_anime_real_model
        _open_anime_real_model.cache_clear()

    def run(self):
        image_file = random.choice(self.all_images)
        _ = anime_real(image_file, self.model)


if __name__ == '__main__':
    create_plot_cli(
        [
            (name, AnimeRealBenchmark(name))
            for name in _MODEL_NAMES
        ],
        title='Benchmark for Anime Real Check Models',
        run_times=10,
        try_times=20,
    )()
