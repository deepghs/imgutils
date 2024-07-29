import random

from benchmark import BaseBenchmark, create_plot_cli
from imgutils.tagging import get_deepdanbooru_tags, get_wd14_tags, get_mldanbooru_tags


class DeepdanbooruBenchmark(BaseBenchmark):
    def load(self):
        from imgutils.tagging.deepdanbooru import _get_deepdanbooru_model
        _ = _get_deepdanbooru_model()

    def unload(self):
        from imgutils.tagging.deepdanbooru import _get_deepdanbooru_model
        _get_deepdanbooru_model.cache_clear()

    def run(self):
        image_file = random.choice(self.all_images)
        _ = get_deepdanbooru_tags(image_file)


class Wd14Benchmark(BaseBenchmark):
    def __init__(self, model):
        BaseBenchmark.__init__(self)
        self.model = model

    def load(self):
        from imgutils.tagging.wd14 import _get_wd14_model
        _ = _get_wd14_model(self.model)

    def unload(self):
        from imgutils.tagging.wd14 import _get_wd14_model
        _get_wd14_model.cache_clear()

    def run(self):
        image_file = random.choice(self.all_images)
        _ = get_wd14_tags(image_file, model_name=self.model)


class MLDanbooruBenchmark(BaseBenchmark):
    def load(self):
        from imgutils.tagging.mldanbooru import _open_mldanbooru_model
        _ = _open_mldanbooru_model()

    def unload(self):
        from imgutils.tagging.mldanbooru import _open_mldanbooru_model
        _open_mldanbooru_model.cache_clear()

    def run(self):
        image_file = random.choice(self.all_images)
        _ = get_mldanbooru_tags(image_file)


if __name__ == '__main__':
    create_plot_cli(
        [
            ('deepdanbooru', DeepdanbooruBenchmark()),
            ('wd14-swinv2', Wd14Benchmark("SwinV2")),
            ('wd14-convnext', Wd14Benchmark("ConvNext")),
            ('wd14-convnextv2', Wd14Benchmark("ConvNextV2")),
            ('wd14-vit', Wd14Benchmark("ViT")),
            ('wd14-moat', Wd14Benchmark("MOAT")),
            ('wd-swinv2-v3', Wd14Benchmark("SwinV2_v3")),
            ('wd-vit-v3', Wd14Benchmark("ViT_v3")),
            ('wd-convnext-v3', Wd14Benchmark("ConvNext_v3")),
            ('wd-vit-large-tagger-v3', Wd14Benchmark("ViT_Large")),
            ('wd-eva02-large-tagger-v3', Wd14Benchmark("EVA02_Large")),
            ('mldanbooru', MLDanbooruBenchmark()),
        ],
        title='Benchmark for Tagging Models',
        run_times=10,
        try_times=20,
    )()
