import pytest

from imgutils.validate import is_truncated_file
from test.testings import get_testfile


@pytest.mark.unittest
class TestValidateTruncate:
    @pytest.mark.parametrize(['filename', 'is_truncated'], [
        ('jpeg_full.jpeg', False),
        ('jpeg_truncated.jpeg', True),
        ('png_full.png', False),
        ('png_truncated.png', True),
    ])
    def test_is_truncated_file(self, filename, is_truncated):
        assert is_truncated_file(get_testfile(filename)) == is_truncated
