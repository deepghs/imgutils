import os
import random

from huggingface_hub import HfFileSystem

from benchmark import BaseBenchmark, create_plot_cli
from imgutils.validate import anime_portrait

hf_fs = HfFileSystem()

_REPOSITORY = 'deepghs/anime_portrait'
_MODEL_NAMES = [
    os.path.relpath(file, _REPOSITORY).split('/')[0] for file in
    hf_fs.glob(f'{_REPOSITORY}/*/model.onnx')
]


class AnimePortraitBenchmark(BaseBenchmark):
    def __init__(self, model):
        BaseBenchmark.__init__(self)
        self.model = model

    def load(self):
        from imgutils.validate.portrait import _open_anime_portrait_model
        _ = _open_anime_portrait_model(self.model)

    def unload(self):
        from imgutils.validate.portrait import _open_anime_portrait_model
        _open_anime_portrait_model.cache_clear()

    def run(self):
        image_file = random.choice(self.all_images)
        _ = anime_portrait(image_file, self.model)


if __name__ == '__main__':
    create_plot_cli(
        [
            (name, AnimePortraitBenchmark(name))
            for name in _MODEL_NAMES
        ],
        title='Benchmark for Anime Portrait Models',
        run_times=10,
        try_times=20,
    )()
