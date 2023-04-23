import pytest

from imgutils.tagging import get_deepdanbooru_tags
from test.testings import get_testfile


@pytest.mark.unittest
class TestTaggingDeepdanbooru:
    def test_get_deepdanbooru_tags(self):
        rating, tags, chars = get_deepdanbooru_tags(get_testfile('6124220.jpg'))
        assert rating['rating:safe'] > 0.9
        assert tags['greyscale'] >= 0.8
        assert tags['pixel_art'] >= 0.9
        assert not chars

        rating, tags, chars = get_deepdanbooru_tags(get_testfile('6125785.jpg'))
        assert rating['rating:safe'] > 0.9
        assert tags['1girl'] >= 0.85
        assert tags['ring'] > 0.8
        assert chars['hu_tao_(genshin_impact)'] >= 0.7
