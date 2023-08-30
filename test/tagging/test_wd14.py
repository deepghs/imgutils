import pytest

from imgutils.tagging import get_wd14_tags
from imgutils.tagging.wd14 import _get_wd14_model
from test.testings import get_testfile


@pytest.fixture()
def _release_model_after_run():
    try:
        yield
    finally:
        _get_wd14_model.cache_clear()


@pytest.mark.unittest
class TestTaggingWd14:
    def test_get_wd14_tags(self):
        rating, tags, chars = get_wd14_tags(get_testfile('6124220.jpg'))
        assert rating['general'] > 0.9
        assert tags['cat'] >= 0.8
        assert not chars

        rating, tags, chars = get_wd14_tags(get_testfile('6125785.jpg'))
        assert 0.55 <= rating['general'] <= 0.65
        assert 0.35 <= rating['sensitive'] <= 0.45
        assert tags['1girl'] >= 0.95
        assert chars['hu_tao_(genshin_impact)'] >= 0.95
