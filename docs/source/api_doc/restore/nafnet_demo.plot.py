import os
from typing import List

from imgutils.restore.nafnet import NafNetModelTyping, restore_with_nafnet
from plot import image_plot

sample_dir = 'sample'

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
        src_file = os.path.join(sample_dir, file)
        items.append((src_file, title))
        for model in models:
            items.append((restore_with_nafnet(src_file, model=model), f'{title}\n(Fixed By {model})'))

    image_plot(
        *items,
        columns=len(models) + 1,
        figsize=(10, 12),
    )
