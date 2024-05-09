import random

from benchmark import BaseBenchmark, create_plot_cli
from imgutils.restore.adversarial import remove_adversarial_noise


class AdversarialRemovalBenchmark(BaseBenchmark):
    def load(self):
        pass

    def unload(self):
        pass

    def run(self):
        image_file = random.choice(self.all_images)

        _ = remove_adversarial_noise(image_file)


if __name__ == '__main__':
    create_plot_cli(
        [
            ('Adversarial Removal', AdversarialRemovalBenchmark()),
        ],
        title='Benchmark for Adversarial Removal Algorithm',
        run_times=5,
        try_times=10,
    )()
