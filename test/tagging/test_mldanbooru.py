import pytest

from imgutils.tagging import get_mldanbooru_tags
from imgutils.tagging.mldanbooru import _open_mldanbooru_model
from test.testings import get_testfile


@pytest.fixture()
def _release_model_after_run():
    try:
        yield
    finally:
        _open_mldanbooru_model.cache_clear()


@pytest.mark.unittest
class TestTaggingmldanbooru:
    @pytest.mark.parametrize(['keep_ratio'], [(True,), (False,)])
    def test_get_mldanbooru_tags(self, keep_ratio):
        tags = get_mldanbooru_tags(get_testfile('6124220.jpg'), keep_ratio=keep_ratio)
        assert tags['cat'] >= 0.8

        tags = get_mldanbooru_tags(get_testfile('6125785.jpg'), keep_ratio=keep_ratio)
        assert tags['1girl'] >= 0.95
