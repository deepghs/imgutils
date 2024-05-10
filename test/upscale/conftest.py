import pytest

from imgutils.data import load_image
from test.testings import get_testfile


@pytest.fixture()
def sample_image():
    yield load_image(get_testfile('surtr_logo.png'), mode='RGB', force_background='white')


@pytest.fixture()
def sample_image_small(sample_image):
    yield sample_image.resize((127, 126))


@pytest.fixture()
def sample_rgba_image():
    yield load_image(get_testfile('rgba_upscale.png'), mode='RGBA', force_background=None)


@pytest.fixture()
def sample_rgba_image_4x():
    yield load_image(get_testfile('rgba_upscale_4x.png'), mode='RGBA', force_background=None)
