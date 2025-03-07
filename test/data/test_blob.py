import pytest
from hbutils.testing import tmatrix

from imgutils.data import is_valid_image_blob_url, load_image, to_blob_url, load_image_from_blob_url
from test.testings import get_testfile


@pytest.mark.unittest
class TestDataBlob:
    @pytest.mark.parametrize(*tmatrix({
        'filename': [
            'mostima_post.jpg',
            'soldiers.jpg',
            'nian.png',
        ],
        'format': [
            'jpg',
            'png',
            'webp',
        ]
    }, mode='matrix'))
    def test_to_blob_url_and_load_image_from_blob_url(self, filename, format, image_diff):
        original_image = load_image(get_testfile(filename), mode='RGB', force_background='white')
        assert image_diff(
            original_image,
            load_image_from_blob_url(to_blob_url(original_image, format=format)),
            throw_exception=False,
        ) < 1.5e-2

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
