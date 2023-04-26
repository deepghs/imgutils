import pytest
from PIL import Image

from imgutils.data import grid_background, load_image, grid_transparent
from ..testings import get_testfile


@pytest.mark.unittest
class TestDataBackground:
    def test_grid_background(self, image_diff):
        image = grid_background(768, 512)
        assert isinstance(image, Image.Image)
        assert image.height == 768
        assert image.width == 512
        assert image.mode == 'RGBA'

        assert image_diff(
            load_image(get_testfile('transparent_bg.png'), mode='RGB'), image.convert('RGB'),
            throw_exception=False
        ) < 1e-2

    @pytest.mark.parametrize(['original_image', 'tp_image'], [
        ('dori.png', 'dori_tp.png'),
        ('nian.png', 'nian_tp.png'),
    ])
    def test_grid_transparent(self, original_image, tp_image, image_diff):
        image = load_image(get_testfile(original_image), force_background=None, mode='RGBA')
        new_image = grid_transparent(image)
        assert new_image.size == image.size
        assert new_image.mode == 'RGB'

        assert image_diff(
            load_image(get_testfile(tp_image), mode='RGB'), new_image,
            throw_exception=False
        ) < 1e-2
