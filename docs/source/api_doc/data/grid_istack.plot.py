import numpy as np
from PIL import Image

from imgutils.data import istack
from plot import image_plot

if __name__ == '__main__':
    width, height = Image.open('nian.png').size
    hs1 = (1 - np.abs(np.linspace(-1 / 3, 1, height))) ** 0.5
    ws1 = (1 - np.abs(np.linspace(-1, 1, width))) ** 0.5
    nian_mask = hs1[..., None] * ws1  # HxW

    image_plot(
        'nian.png',
        (istack('lime', 'nian.png'), 'nian_lime.png'),
        (istack(('yellow', 0.5), ('nian.png', 0.9)), 'nian_trans.png'),
        (istack(('nian.png', nian_mask)), 'nian_mask.png'),
        columns=2,
        figsize=(12, 12),
    )
