import os.path

import pytest
from hbutils.testing import tmatrix

from imgutils.data import load_image
from imgutils.segment import segment_with_isnetis, segment_rgba_with_isnetis
from ..testings import get_testfile


@pytest.mark.unittest
class TestSegmentIsnetis:
    @pytest.mark.parametrize(*tmatrix({
        'original_image': ['6125785.jpg', '6125901.jpg'],
        'background': ['', 'black', 'white'],
    }))
    def test_segment_with_isnetis(self, original_image, background, image_diff):
        image = load_image(get_testfile(original_image))
        obody, oext = os.path.splitext(original_image)
        target_image_file = get_testfile(f'isnetis_{obody}_{background}{oext}')

        mask, ret_image = segment_with_isnetis(image, **({'background': background} if background else {}))
        assert mask.shape == (image.height, image.width)
        assert ret_image.size == image.size
        assert ret_image.mode == 'RGB'

        assert image_diff(
            load_image(target_image_file, mode='RGB'), ret_image,
            throw_exception=False
        ) < 1e-2

    @pytest.mark.parametrize(*tmatrix({
        'original_image': ['6125785.jpg', '6125901.jpg'],
    }))
    def test_segment_rgba_with_isnetis(self, original_image, image_diff):
        image = load_image(get_testfile(original_image))
        obody, oext = os.path.splitext(original_image)
        target_image_file = get_testfile(f'isnetis_rgba_{obody}.png')

        mask, ret_image = segment_rgba_with_isnetis(image)
        assert mask.shape == (image.height, image.width)
        assert ret_image.size == image.size
        assert ret_image.mode == 'RGBA'

        for color in ['red', 'green', 'white', 'black', 'pink']:
            assert image_diff(
                load_image(target_image_file, force_background=color, mode='RGB'),
                load_image(ret_image, force_background=color, mode='RGB'),
                throw_exception=False
            ) < 1e-2
