import pytest

from imgutils.tagging.match import tag_match_suffix, tag_match_prefix, tag_match_full


@pytest.mark.unittest
class TestTaggingMatch:
    def test_tag_match_suffix(self):
        assert tag_match_suffix('ears', 'ear')
        assert tag_match_suffix('ear', 'ears')
        assert tag_match_suffix('cat ears', 'ear')
        assert tag_match_suffix('cat_ear', 'ears')
        assert tag_match_suffix('cat_ear', 'cat ears')
        assert not tag_match_suffix('cat tails', 'ear')
        assert not tag_match_suffix('cat tail', 'ears')
        assert tag_match_suffix('red_cat ears', 'cat_ear')
        assert not tag_match_suffix('red cats ear', 'cat ear')
        assert tag_match_suffix('ears', '')
        assert tag_match_suffix('cat ear', '')
        assert tag_match_suffix('', '')

    def test_tag_match_prefix(self):
        assert tag_match_prefix('cat ears', 'cat')
        assert tag_match_prefix('cat_ears', 'cat')
        assert not tag_match_suffix('cats_ears', 'cat')
        assert tag_match_prefix('cat ears', '')
        assert tag_match_prefix('cats ears', '')

    def test_tag_match_full(self):
        assert tag_match_full('cat_ear', 'cat ears')
        assert tag_match_full('cat_ears', 'cat ear')
        assert tag_match_full('', '')
        assert tag_match_full('ears', 'ear')
        assert tag_match_full('ear', 'ears')
        assert not tag_match_full('cat ears', 'ear')
        assert not tag_match_full('cat_ear', 'ears')
        assert not tag_match_full('ears', '')
        assert not tag_match_full('ear', '')
        assert not tag_match_full('cat ears', '')
        assert not tag_match_full('cat_ear', '')
