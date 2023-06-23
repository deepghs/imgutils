import numpy as np
from PIL import Image

from imgutils.operate import squeeze
from plot import image_plot

if __name__ == '__main__':
    image = Image.open('jerry_with_space.png')

    mask = np.array(image)[:, :, 3]
    mask_image = Image.fromarray(mask, 'L')

    image_plot(
        (image, 'origin'),
        (mask_image, 'mask (numpy bool[H, W])'),
        (squeeze(image, mask), 'squeezed'),
        columns=3,
        figsize=(10, 4),
        keep_axis=True,
    )
