import os
import random

from huggingface_hub import HfFileSystem

from benchmark import BaseBenchmark, create_plot_cli
from imgutils.validate import anime_bangumi_char

hf_fs = HfFileSystem()

_REPOSITORY = 'deepghs/bangumi_char_type'
_MODEL_NAMES = [
    os.path.relpath(file, _REPOSITORY).split('/')[0] for file in
    hf_fs.glob(f'{_REPOSITORY}/*/model.onnx')
]


class AnimeBangumiCharacterBenchmark(BaseBenchmark):
    def __init__(self, model):
        BaseBenchmark.__init__(self)
        self.model = model

    def load(self):
        from imgutils.validate.bangumi_char import _open_anime_bangumi_char_model
        _ = _open_anime_bangumi_char_model(self.model)

    def unload(self):
        from imgutils.validate.bangumi_char import _open_anime_bangumi_char_model
        _open_anime_bangumi_char_model.cache_clear()

    def run(self):
        image_file = random.choice(self.all_images)
        _ = anime_bangumi_char(image_file, self.model)


if __name__ == '__main__':
    create_plot_cli(
        [
            (name, AnimeBangumiCharacterBenchmark(name))
            for name in _MODEL_NAMES
        ],
        title='Benchmark for Bangumi Character Type Models',
        run_times=10,
        try_times=20,
    )()
