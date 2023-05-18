import random

import numpy as np
from PIL import Image

from plot import image_plot

ORIGIN_FILENAME = 'psnr/origin.jpg'


def gaussian_noise(scale=20):
    random.seed(0)
    img = Image.open(ORIGIN_FILENAME).convert('RGB')
    img_arr = np.array(img)
    noise = np.random.normal(0, scale, img_arr.shape)
    noisy_arr = img_arr + noise
    noisy_img = Image.fromarray(noisy_arr.astype('uint8'), mode='RGB')

    export_file = f'psnr/gaussian_{scale}.dat.jpg'
    noisy_img.save(export_file)
    return export_file


def low_quality():
    random.seed(0)
    img = Image.open(ORIGIN_FILENAME).convert('RGB')
    export_file = 'psnr/lq.dat.jpg'
    img.save(export_file, quality=5)
    return export_file


if __name__ == '__main__':
    image_plot(
        ORIGIN_FILENAME,
        gaussian_noise(20),
        gaussian_noise(3),
        low_quality(),
        columns=2,
        figsize=(6, 5)
    )
