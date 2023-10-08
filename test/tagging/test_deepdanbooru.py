import pytest

from imgutils.tagging import get_deepdanbooru_tags
from imgutils.tagging.deepdanbooru import _get_deepdanbooru_model
from test.testings import get_testfile


@pytest.fixture()
def _release_model_after_run():
    try:
        yield
    finally:
        _get_deepdanbooru_model.cache_clear()


@pytest.mark.unittest
class TestTaggingDeepdanbooru:
    def test_get_deepdanbooru_tags(self, _release_model_after_run):
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

    def test_get_danbooru_tags_sample(self):
        rating, tags, chars = get_deepdanbooru_tags(get_testfile('nude_girl.png'))
        assert rating == pytest.approx({
            'rating:safe': 8.940696716308594e-06,
            'rating:questionable': 0.012878596782684326,
            'rating:explicit': 0.992286205291748,
        }, abs=1e-3)
        assert tags == pytest.approx({
            '1girl': 0.9923416376113892, 'armpits': 0.9226008653640747, 'arms_behind_head': 0.5620371699333191,
            'arms_up': 0.7268614172935486, 'bangs': 0.7465004920959473, 'black_border': 0.9081975221633911,
            'blush': 0.9306209683418274, 'breasts': 0.9972158670425415,
            'eyebrows_visible_through_hair': 0.6717097163200378, 'hair_between_eyes': 0.7044132947921753,
            'hair_intakes': 0.6295598745346069, 'horns': 0.9387356042861938, 'letterboxed': 1.0,
            'long_hair': 0.9871174693107605, 'looking_at_viewer': 0.8953969478607178,
            'medium_breasts': 0.90318363904953, 'navel': 0.9425054788589478, 'nipples': 0.9989081621170044,
            'nude': 0.9452821016311646, 'pillarboxed': 0.9854832887649536, 'purple_eyes': 0.8120401501655579,
            'pussy': 0.9943721294403076, 'pussy_juice': 0.8238061666488647, 'red_hair': 0.9203640222549438,
            'smile': 0.6659414172172546, 'solo': 0.9483305811882019, 'spread_legs': 0.7633067965507507,
            'stomach': 0.5396291017532349, 'sweat': 0.7880321145057678, 'thighs': 0.7451953291893005,
            'uncensored': 0.9594683647155762, 'very_long_hair': 0.740706205368042,
        }, abs=1e-3)
        assert chars == pytest.approx({'surtr_(arknights)': 0.9373699426651001}, abs=1e-3)

    def test_get_danbooru_tags_drop_overlap(self):
        rating, tags, chars = get_deepdanbooru_tags(get_testfile('nude_girl.png'), drop_overlap=True)
        assert rating == pytest.approx({
            'rating:safe': 8.940696716308594e-06,
            'rating:questionable': 0.012878596782684326,
            'rating:explicit': 0.992286205291748,
        }, abs=1e-3)
        assert tags == pytest.approx({
            '1girl': 0.9923416376113892, 'armpits': 0.9226007461547852, 'arms_behind_head': 0.5620364546775818,
            'arms_up': 0.7268615365028381, 'bangs': 0.7465004324913025, 'black_border': 0.9081975221633911,
            'blush': 0.9306209683418274, 'eyebrows_visible_through_hair': 0.6717095971107483,
            'hair_between_eyes': 0.7044129967689514, 'hair_intakes': 0.6295579671859741, 'horns': 0.938735842704773,
            'letterboxed': 1.0, 'looking_at_viewer': 0.8953973650932312, 'medium_breasts': 0.9031840562820435,
            'navel': 0.9425054788589478, 'nipples': 0.9989081621170044, 'nude': 0.9452821016311646,
            'pillarboxed': 0.9854832887649536, 'purple_eyes': 0.8120403289794922, 'pussy_juice': 0.8238056898117065,
            'red_hair': 0.9203639030456543, 'smile': 0.6659414172172546, 'solo': 0.948330819606781,
            'spread_legs': 0.7633066177368164, 'stomach': 0.5396295189857483, 'sweat': 0.7880324721336365,
            'thighs': 0.745195746421814, 'uncensored': 0.9594683647155762, 'very_long_hair': 0.7407056093215942
        }, abs=1e-3)
        assert chars == pytest.approx({'surtr_(arknights)': 0.9373699426651001}, abs=1e-3)
