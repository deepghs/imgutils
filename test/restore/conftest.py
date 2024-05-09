import os.path

import numpy as np
import pytest
from PIL import Image
from hbutils.system import TemporaryDirectory

from imgutils.data import load_image
from test.testings import get_testfile


@pytest.fixture()
def sample_image():
    yield load_image(get_testfile('surtr_logo.png'), mode='RGB', force_background='white')


@pytest.fixture()
def clear_image():
    yield load_image(get_testfile('surtr_logo_clear.png'), mode='RGB', force_background='white')


def add_gaussian_noise(image, mean=0, std=25):
    img_array = np.array(image)
    noise = np.random.normal(mean, std, img_array.shape)
    noisy_image = img_array + noise
    noisy_image = np.clip(noisy_image, 0, 255)
    noisy_image = Image.fromarray(np.uint8(noisy_image))
    return noisy_image


@pytest.fixture()
def gaussian_noise_image(sample_image):
    yield add_gaussian_noise(sample_image)


@pytest.fixture()
def q45_image(sample_image):
    with TemporaryDirectory() as td:
        img_file = os.path.join(td, 'image.jpg')
        sample_image.save(img_file, quality=45)
        yield load_image(img_file)


@pytest.fixture()
def rgba_image():
    yield load_image(get_testfile('rgba_restore.png'), mode='RGBA', force_background=None)
