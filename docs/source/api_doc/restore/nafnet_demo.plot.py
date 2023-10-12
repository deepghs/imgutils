import os
from typing import List

from imgutils.restore.nafnet import NafNetModelTyping
from plot import image_plot

sample_dir = 'sample'
nafnet_dir = 'nafnet'

if __name__ == '__main__':
    models: List[NafNetModelTyping] = ['REDS', 'GoPro', 'SIDD']
    demo_images = os.listdir(sample_dir)

    items = []
    for file in demo_images:
        items.append((os.path.join(sample_dir, file), file))
        for model in models:
            dst_file = os.path.join(nafnet_dir, f'{os.path.splitext(file)[0]}-{model.lower()}.png')
            items.append((dst_file, os.path.basename(dst_file)))

    image_plot(
        *items,
        columns=len(models) + 1,
        figsize=(14, 7),
    )
