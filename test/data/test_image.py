import pytest
from PIL import Image

from imgutils.data import load_image
from test.testings import get_testfile

_FILENAME = get_testfile('6125785.png')
_IMAGE = Image.open(_FILENAME)


@pytest.mark.unittest
class TestDataImage:
    @pytest.mark.parametrize(['image_', 'result'], [
        (_FILENAME, _IMAGE),
        (_IMAGE, _IMAGE),
        (None, TypeError),
    ])
    def test_load_image(self, image_, result, image_diff):
        if isinstance(result, type) and issubclass(result, BaseException):
            with pytest.raises(result):
                _ = load_image(image_)
        elif isinstance(image_, Image.Image):
            assert load_image(image_, force_background=None) is image_
        else:
            assert image_diff(load_image(image_), result, throw_exception=False) < 1e-2
