import os

from imgutils.data import load_image
from imgutils.restore import remove_adversarial_noise
from plot import image_plot

sample_dir = 'sample'

if __name__ == '__main__':
    image = load_image(os.path.join(sample_dir, 'adversarial_input.png'))
    image_plot(
        [
            (image, 'Adversarial Noised'),
            (remove_adversarial_noise(image), 'Cleaned')
        ],
        columns=2,
        figsize=(10, 6),
    )
