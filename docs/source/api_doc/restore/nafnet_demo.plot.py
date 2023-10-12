import os
from typing import List

from imgutils.restore.nafnet import NafNetModelTyping
from plot import image_plot

sample_dir = 'sample'
nafnet_dir = 'nafnet'

if __name__ == '__main__':
    models: List[NafNetModelTyping] = ['REDS', 'GoPro', 'SIDD']
    demo_images = [
        ('original.png', 'Original'),
        ('blur.png', 'Blur'),
        ('jpg-q45.jpg', 'JPEG Quality45'),
        ('jpg-q35.jpg', 'JPEG Quality35'),
    ]

    items = []
    for file, title in demo_images:
        items.append((os.path.join(sample_dir, file), title))
        for model in models:
            dst_file = os.path.join(nafnet_dir, f'{os.path.splitext(file)[0]}-{model.lower()}.png')
            items.append((dst_file, f'{title}\n(Fixed By {model})'))

    image_plot(
        *items,
        columns=len(models) + 1,
        figsize=(10, 12.5),
    )
