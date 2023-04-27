import numpy as np
import pytest
from PIL import Image
from hbutils.testing import tmatrix

from imgutils.data import istack, load_image
from ..testings import get_testfile


@pytest.mark.unittest
class TestDataLayer:
    def test_istack(self, image_diff):
        width, height = Image.open(get_testfile('nian.png')).size
        hs1 = (1 - np.abs(np.linspace(-1, 1, height))) ** 0.5
        ws1 = (1 - np.abs(np.linspace(-1, 1, width))) ** 0.5
        nian_mask = hs1[..., None] * ws1

        hs2 = np.abs(np.linspace(-1, 1, height)) ** 0.5
        ws2 = np.abs(np.linspace(-1, 1, width)) ** 0.5
        color_mask = hs2[..., None] * ws2

        image = istack((get_testfile('nian.png'), nian_mask), ('green', color_mask))
        for color in ['red', 'green', 'white', 'black', 'pink']:
            assert image_diff(
                load_image(get_testfile('nian_green.png'), force_background=color, mode='RGB'),
                load_image(image, force_background=color, mode='RGB'),
                throw_exception=False
            ) < 1e-2

    @pytest.mark.parametrize(*tmatrix({
        'color1': ['red', 'green', 'blue'],
        'color2': ['red', 'green', 'blue'],
    }))
    def test_istack_pure_color(self, color1, color2, image_diff):
        image = istack((color1, 0.7), (color2, 0.4), size=(768, 512))
        target_filename = get_testfile(f'istack_{color1}_{color2}.png')

        assert image.height == 512
        assert image.width == 768
        assert image.mode == 'RGBA'

        for color in ['red', 'green', 'white', 'black', 'pink']:
            assert image_diff(
                load_image(target_filename, force_background=color, mode='RGB'),
                load_image(image, force_background=color, mode='RGB'),
                throw_exception=False
            ) < 1e-2

    def test_istack_error(self):
        with pytest.raises(ValueError):
            _ = istack(('red', 0.5), ('green', 0.5))
