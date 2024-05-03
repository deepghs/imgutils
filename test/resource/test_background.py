import os.path

import pytest
from PIL import Image

from imgutils.resource import list_bg_image_files, get_bg_image_file, get_bg_image, random_bg_image_file, \
    random_bg_image, BackgroundImageSet


@pytest.fixture(scope='module')
def set_size_3000():
    return BackgroundImageSet(min_width=3000, min_height=3000)


@pytest.fixture(scope='module')
def set_width_1200():
    return BackgroundImageSet(width=1200)


@pytest.fixture(scope='module')
def set_height_1200():
    return BackgroundImageSet(height=1200)


@pytest.fixture(scope='module')
def set_resolution_2000():
    return BackgroundImageSet(min_resolution=2000)


@pytest.fixture(scope='module')
def set_size_10_2():
    return BackgroundImageSet(width=10, height=2, strict_level=4)


@pytest.mark.unittest
class TestResourceBackground:
    def test_list_bg_image_files(self):
        files = list_bg_image_files()
        assert len(files) == 8057

    @pytest.mark.parametrize(['filename'], [
        (f'{i:06d}.jpg',) for i in range(10)
    ])
    def test_get_bg_image_file(self, filename):
        file = get_bg_image_file(filename)
        assert os.path.exists(file)

    def test_get_bg_image_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            _ = get_bg_image_file('file_not_found.jpg')

    @pytest.mark.parametrize(['filename'], [
        (f'{i:06d}.jpg',) for i in range(10)
    ])
    def test_get_bg_image(self, filename):
        image = get_bg_image(filename)
        assert isinstance(image, Image.Image)
        assert image.width >= 1000
        assert image.height >= 1000

    def test_get_bg_image_not_found(self):
        with pytest.raises(FileNotFoundError):
            _ = get_bg_image('file_not_found.jpg')

    @pytest.mark.parametrize(['i'], [
        (i,) for i in range(3)
    ])
    def test_random_bg_image_file(self, i):
        file = random_bg_image_file()
        assert os.path.exists(file)

    @pytest.mark.parametrize(['i'], [
        (i,) for i in range(3)
    ])
    def test_random_bg_image(self, i):
        image = random_bg_image()
        assert isinstance(image, Image.Image)

    def test_min_size(self, set_size_3000):
        assert len(set_size_3000.list_image_files()) < 200
        assert os.path.exists(set_size_3000.get_image_file('000091.jpg'))
        with pytest.raises(FileNotFoundError):
            set_size_3000.get_image_file('008054.jpg')

    def test_x_size(self):
        with pytest.raises(ValueError):
            _ = BackgroundImageSet(min_width=30000, min_height=30000)

    def test_width(self, set_width_1200):
        for _ in range(5):
            image = set_width_1200.random_image()
            assert 1100 <= image.width <= 1300

    def test_height(self, set_height_1200):
        for _ in range(5):
            image = set_height_1200.random_image()
            assert 1100 <= image.height <= 1300

    def test_resolution(self, set_resolution_2000):
        for _ in range(5):
            image = set_resolution_2000.random_image()
            assert image.width * image.height >= 2000 ** 2

    def test_ratio(self, set_size_10_2):
        assert 10 <= len(set_size_10_2.df) <= 20
        v = set_size_10_2.df['width'] / set_size_10_2.df['height']
        assert ((v >= 4) & (v <= 6)).all()
