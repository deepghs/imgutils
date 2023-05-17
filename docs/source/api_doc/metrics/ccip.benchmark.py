import random

from benchmark import BaseBenchmark, create_plot_cli
from imgutils.metrics.ccip import get_ccip_features, get_ccip_similarity, _VALID_MODEL_NAMES


class CCIPFeatureBenchmark(BaseBenchmark):
    def __init__(self, model_name):
        BaseBenchmark.__init__(self)
        self.model_name = model_name

    def load(self):
        from imgutils.metrics.ccip import _open_feat_model
        _ = _open_feat_model(self.model_name)

    def unload(self):
        from imgutils.metrics.ccip import _open_feat_model
        _open_feat_model.cache_clear()

    def run(self):
        image_file = random.choice(self.all_images)
        _ = get_ccip_features([image_file], model_name=self.model_name)


class CCIPDiffBenchmark(BaseBenchmark):
    def __init__(self, model_name):
        BaseBenchmark.__init__(self)
        self.model_name = model_name

    def prepare(self):
        self.feats = list(get_ccip_features(random.sample(self.all_images, k=30), model_name=self.model_name))

    def load(self):
        from imgutils.metrics.ccip import _open_metric_model
        _ = _open_metric_model(self.model_name)

    def unload(self):
        from imgutils.metrics.ccip import _open_metric_model
        _open_metric_model.cache_clear()

    def run(self):
        feat1 = random.choice(self.feats)
        feat2 = random.choice(self.feats)
        _ = get_ccip_similarity(feat1, feat2, model_name=self.model_name)


if __name__ == '__main__':
    bms = []
    for model_name in _VALID_MODEL_NAMES:
        bms.append((f'{model_name} extract', CCIPFeatureBenchmark(model_name)))
        bms.append((f'{model_name} metrics', CCIPDiffBenchmark(model_name)))

    create_plot_cli(
        bms,
        title='Benchmark for CCIP Models',
        run_times=10,
        try_times=20,
    )()
