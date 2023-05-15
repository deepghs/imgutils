import random

from benchmark import BaseBenchmark, create_plot_cli
from imgutils.metrics import lpips_extract_feature, lpips_difference


class LPIPSFeatureBenchmark(BaseBenchmark):
    def load(self):
        from imgutils.metrics.lpips import _lpips_feature_model
        _ = _lpips_feature_model()

    def unload(self):
        from imgutils.metrics.lpips import _lpips_feature_model
        _lpips_feature_model.cache_clear()

    def run(self):
        image_file = random.choice(self.all_images)
        _ = lpips_extract_feature(image_file)


class LPIPSDiffBenchmark(BaseBenchmark):
    def prepare(self):
        self.feats = [lpips_extract_feature(img) for img in random.sample(self.all_images, k=30)]

    def load(self):
        from imgutils.metrics.lpips import _lpips_diff_model
        _ = _lpips_diff_model()

    def unload(self):
        from imgutils.metrics.lpips import _lpips_diff_model
        _lpips_diff_model.cache_clear()

    def run(self):
        feat1 = random.choice(self.feats)
        feat2 = random.choice(self.feats)
        _ = lpips_difference(feat1, feat2)


if __name__ == '__main__':
    create_plot_cli(
        [
            ('feature extract', LPIPSFeatureBenchmark()),
            ('diff calculate', LPIPSDiffBenchmark()),
        ],
        title='Benchmark for LPIPS Models',
        run_times=10,
        try_times=20,
    )()
