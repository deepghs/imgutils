import pytest

from imgutils.data import is_valid_image_blob_url


@pytest.mark.unittest
class TestDataBlob:
    @pytest.mark.parametrize(['blob_url', 'expected_result'], [
        ("data:image/png;base64,ABC", True),
        ("DATA:IMAGE/JPEG,", True),
        ("data:image/svg+xml;charset=utf-8,<svg/>", True),
        ("data:image/webp;param=1;param2=2,data", True),
        ("data:text/plain,Hello", False),
        ("data:image/png", False),
        ("data:image/;base64,ABC", False),
        ("data:video/mp4;base64,AAA", False),
        ("data:image", False),
    ])
    def test_is_valid_image_blob_url(self, blob_url, expected_result):
        assert is_valid_image_blob_url(blob_url) == expected_result
