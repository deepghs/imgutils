import pytest

from imgutils.validate import is_truncated_file
from test.testings import get_testfile


@pytest.fixture()
def jpeg_full():
    return get_testfile('jpeg_full.jpeg')


@pytest.fixture()
def jpeg_truncated():
    return get_testfile('jpeg_truncated.jpeg')


@pytest.fixture()
def png_full():
    return get_testfile('png_full.png')


@pytest.fixture()
def png_truncated():
    return get_testfile('png_truncated.png')


@pytest.mark.unittest
class TestValidateTruncate:
    def test_is_truncated_file(self, jpeg_full, jpeg_truncated, png_full, png_truncated):
        assert is_truncated_file(jpeg_truncated)
        assert not is_truncated_file(jpeg_full)
        assert is_truncated_file(png_truncated)
        assert not is_truncated_file(png_full)
