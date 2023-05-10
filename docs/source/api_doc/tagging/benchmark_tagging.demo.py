import random

from benchmark import BaseBenchmark, create_plot
from imgutils.tagging import get_deepdanbooru_tags, get_wd14_tags


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


if __name__ == '__main__':
    create_plot(
        [
            ('deepdanbooru', DeepdanbooruBenchmark()),
            ('wd14-swinv2', Wd14Benchmark("SwinV2")),
            ('wd14-convnext', Wd14Benchmark("ConvNext")),
            ('wd14-convnextv2', Wd14Benchmark("ConvNextV2")),
            ('wd14-vit', Wd14Benchmark("ViT")),
        ],
        save_as='benchmark_tagging.bm.svg',
        title='Benchmark for Tagging Models',
        run_times=10,
        try_times=5,
        figsize=(1080, 600)
    )
