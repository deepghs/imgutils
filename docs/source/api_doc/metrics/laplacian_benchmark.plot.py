import random

from benchmark import BaseBenchmark, create_plot_cli
from imgutils.metrics import laplacian_score


class LaplacianBenchmark(BaseBenchmark):
    def load(self):
        pass

    def unload(self):
        pass

    def run(self):
        image_file = random.choice(self.all_images)
        _ = laplacian_score(image_file)


if __name__ == '__main__':
    create_plot_cli(
        [
            ('Laplacian', LaplacianBenchmark())
        ],
        title='Benchmark for Laplacian Blur Models',
        run_times=10,
        try_times=20,
    )()
