import random

from benchmark import BaseBenchmark, create_plot_cli
from imgutils.tagging import get_deepdanbooru_tags, get_wd14_tags, get_mldanbooru_tags, get_deepgelbooru_tags, \
    get_camie_tags


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


class DeepgelbooruBenchmark(BaseBenchmark):
    def load(self):
        from imgutils.tagging.deepgelbooru import _open_tags, _open_model, _open_preprocessor
        _ = _open_tags()
        _ = _open_model()
        _ = _open_preprocessor

    def unload(self):
        from imgutils.tagging.deepgelbooru import _open_tags, _open_model, _open_preprocessor
        _open_tags.cache_clear()
        _open_model.cache_clear()
        _open_preprocessor.cache_clear()

    def run(self):
        image_file = random.choice(self.all_images)
        _ = get_deepgelbooru_tags(image_file)


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


class CamieBenchmark(BaseBenchmark):
    def __init__(self, model_name):
        BaseBenchmark.__init__(self)
        self.model_name = model_name

    def load(self):
        from imgutils.tagging.camie import _get_camie_model, _get_camie_labels, _get_camie_threshold, \
            _get_camie_preprocessor
        _ = _get_camie_model(self.model_name)
        _ = _get_camie_labels(self.model_name)
        _ = _get_camie_threshold(self.model_name)
        _ = _get_camie_preprocessor(self.model_name)

    def unload(self):
        from imgutils.tagging.camie import _get_camie_model, _get_camie_labels, _get_camie_threshold, \
            _get_camie_preprocessor
        _get_camie_model.cache_clear()
        _get_camie_labels.cache_clear()
        _get_camie_threshold.cache_clear()
        _get_camie_preprocessor.cache_clear()

    def run(self):
        image_file = random.choice(self.all_images)
        _ = get_camie_tags(image_file, model_name=self.model_name)


if __name__ == '__main__':
    create_plot_cli(
        [
            ('deepdanbooru', DeepdanbooruBenchmark()),
            ('deepgelbooru', DeepgelbooruBenchmark()),
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
            ('camie-initial', CamieBenchmark('initial')),
            ('camie-refined', CamieBenchmark('refined')),
        ],
        title='Benchmark for Tagging Models',
        run_times=10,
        try_times=20,
    )()
