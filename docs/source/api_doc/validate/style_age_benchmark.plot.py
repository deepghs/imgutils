import os
import random

from huggingface_hub import HfFileSystem

from benchmark import BaseBenchmark, create_plot_cli
from imgutils.validate import anime_style_age

hf_fs = HfFileSystem()

_REPOSITORY = 'deepghs/anime_style_ages'
_MODEL_NAMES = [
    os.path.relpath(file, _REPOSITORY).split('/')[0] for file in
    hf_fs.glob(f'{_REPOSITORY}/*/model.onnx')
]


class AnimeStyleAgeBenchmark(BaseBenchmark):
    def __init__(self, model):
        BaseBenchmark.__init__(self)
        self.model = model

    def load(self):
        from imgutils.validate.style_age import _open_anime_style_age_model
        _ = _open_anime_style_age_model(self.model)

    def unload(self):
        from imgutils.validate.style_age import _open_anime_style_age_model
        _open_anime_style_age_model.cache_clear()

    def run(self):
        image_file = random.choice(self.all_images)
        _ = anime_style_age(image_file, self.model)


if __name__ == '__main__':
    create_plot_cli(
        [
            (name, AnimeStyleAgeBenchmark(name))
            for name in _MODEL_NAMES
        ],
        title='Benchmark for Anime Style Age Models',
        run_times=10,
        try_times=20,
    )()
