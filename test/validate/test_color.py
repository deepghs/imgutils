import pytest

from imgutils.validate import is_greyscale
from test.testings import get_testfile


@pytest.mark.unittest
class TestValidateColor:
    @pytest.mark.parametrize(['filename', 'is_greyscale_'], [
        ('6124220.jpg', True),
        ('6125785.jpg', False),
        ('6125785.png', False),
        ('6125901.jpg', False),
    ])
    def test_is_greyscale(self, filename, is_greyscale_):
        assert is_greyscale(get_testfile(filename)) == is_greyscale_
