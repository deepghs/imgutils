import numpy as np
import pytest
from PIL import Image

from imgutils.data import load_image
from imgutils.operate import squeeze_with_transparency, squeeze
from test.testings import get_testfile


@pytest.fixture()
def jerry_with_space() -> Image.Image:
    return Image.open(get_testfile('jerry_with_space.png'))


@pytest.fixture()
def jerry_mask(jerry_with_space) -> np.ndarray:
    return np.array(jerry_with_space)[:, :, 3] > 0


@pytest.mark.unittest
class TestOperateSqueeze:
    def test_squeeze_with_transparency(self, jerry_with_space, image_diff):
        assert image_diff(
            load_image(squeeze_with_transparency(jerry_with_space)).convert('RGB'),
            load_image(get_testfile('jerry_with_space_squeeze_t.png')).convert('RGB'),
            throw_exception=False,
        ) < 1e-2

    def test_squeeze(self, jerry_with_space, jerry_mask, image_diff):
        assert image_diff(
            load_image(squeeze(jerry_with_space, jerry_mask)).convert('RGB'),
            load_image(get_testfile('jerry_with_space_squeeze.png')).convert('RGB'),
            throw_exception=False,
        ) < 1e-2

    def test_squeeze_fail(self, jerry_with_space, jerry_mask):
        with pytest.raises(ValueError):
            _ = squeeze(jerry_with_space, jerry_mask[:-1, :-1])
